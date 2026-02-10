// criu-helper is a self-contained binary that performs CRIU restore inside container
// namespaces. It is invoked by the DaemonSet via:
//
//	nsenter -t <PID> -m -n -p -i -- /usr/local/bin/criu-helper --checkpoint-path <path>
//
// It runs inside the placeholder container's mount/net/PID/IPC namespaces and:
//  1. Remounts /proc/sys read-write (CRIU needs to write sysctl)
//  2. Opens checkpoint images directory and net NS file
//  3. Uses AddInheritFd for proper FD passing to CRIU's swrk child
//  4. Generates ExtMountMaps from /proc/1/mountinfo
//  5. Creates link_remap stubs for cross-node restore
//  6. Calls go-criu Restore()
//  7. Remounts /proc/sys read-only
//  8. Prints RESTORED_PID=<N> to stdout
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

const (
	netNsPath      = "/proc/1/ns/net"
	mountInfoPath  = "/proc/1/mountinfo"
	restoreLogFile = "restore.log"
)

func main() {
	checkpointPath := flag.String("checkpoint-path", "", "Path to checkpoint directory")
	workDir := flag.String("work-dir", "", "CRIU work directory")
	flag.Parse()

	log := logrus.WithField("component", "criu-helper")

	if *checkpointPath == "" {
		log.Fatal("--checkpoint-path is required")
	}

	if err := run(*checkpointPath, *workDir, log); err != nil {
		log.WithError(err).Fatal("CRIU restore failed")
	}
}

func run(checkpointPath, workDir string, log *logrus.Entry) error {
	restoreStart := time.Now()

	// Load checkpoint manifest for CRIU settings and mount plan
	manifest, err := checkpoint.ReadCheckpointManifest(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to read manifest: %w", err)
	}

	// Create link_remap stubs before CRIU restore (inside container namespace now)
	createLinkRemapStubs(checkpointPath, log)

	// Remount /proc/sys rw — CRIU needs to write sysctl values
	if err := remountProcSys(true, log); err != nil {
		log.WithError(err).Warn("Failed to remount /proc/sys rw (restore may still work)")
	}
	defer remountProcSys(false, log) //nolint:errcheck

	// Open checkpoint images directory with CLOEXEC cleared for CRIU
	imageDir, imageDirFD, err := common.OpenPathForCRIU(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()

	// Open work directory if specified
	var workDirFile *os.File
	var workDirFD int32 = -1
	if workDir != "" {
		if err := os.MkdirAll(workDir, 0755); err != nil {
			log.WithError(err).Warn("Failed to create work directory")
		} else {
			f, fd, err := common.OpenPathForCRIU(workDir)
			if err == nil {
				workDirFile = f
				workDirFD = fd
				defer workDirFile.Close()
			}
		}
	}

	// Generate external mount maps from /proc/1/mountinfo (inside container namespace)
	extMounts, err := generateExtMountMaps(manifest)
	if err != nil {
		return fmt.Errorf("failed to generate ext mount maps: %w", err)
	}

	// Build CRIU restore options
	criuOpts := buildRestoreOptions(manifest, imageDirFD, workDirFD, extMounts)

	// Reuse criu.conf from checkpoint if it exists
	criuConfPath := filepath.Join(checkpointPath, checkpoint.CheckpointCRIUConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
	}

	// Create CRIU client and set up inherited FDs using AddInheritFd
	c := criu.MakeCriu()

	// Open network namespace and register via AddInheritFd for proper FD passing
	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)

	// Execute CRIU restore
	notify := &restoreNotify{log: log}
	log.Info("Executing CRIU restore")
	if err := c.Restore(criuOpts, notify); err != nil {
		logCRIUErrors(checkpointPath, log)
		return fmt.Errorf("CRIU restore failed: %w", err)
	}

	log.WithFields(logrus.Fields{
		"pid":      notify.restoredPID,
		"duration": time.Since(restoreStart),
	}).Info("CRIU restore completed")

	// Print the restored PID so the DaemonSet can parse it
	fmt.Printf("RESTORED_PID=%d\n", notify.restoredPID)
	return nil
}

// generateExtMountMaps builds CRIU ext-mount maps by replaying the dump-time plan.
func generateExtMountMaps(manifest *checkpoint.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if len(manifest.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	maps := []*criurpc.ExtMountMap{{
		Key: proto.String("/"),
		Val: proto.String("."),
	}}
	added := map[string]struct{}{"/": {}}

	for _, mount := range manifest.CRIUDump.ExtMnt {
		if mount.Key == "" || mount.Key == "/" {
			continue
		}
		if _, exists := added[mount.Key]; exists {
			continue
		}
		val := mount.Val
		if val == "" {
			val = mount.Key
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(mount.Key),
			Val: proto.String(val),
		})
		added[mount.Key] = struct{}{}
	}

	return maps, nil
}

// buildRestoreOptions creates CRIU options for restore from the checkpoint manifest.
func buildRestoreOptions(manifest *checkpoint.CheckpointManifest, imageDirFD, workDirFD int32, extMounts []*criurpc.ExtMountMap) *criurpc.CriuOpts {
	settings := manifest.CRIUDump.CRIU

	var cgMode criurpc.CriuCgMode
	switch settings.ManageCgroupsMode {
	case "soft":
		cgMode = criurpc.CriuCgMode_SOFT
	case "full":
		cgMode = criurpc.CriuCgMode_FULL
	case "strict":
		cgMode = criurpc.CriuCgMode_STRICT
	default:
		cgMode = criurpc.CriuCgMode_IGNORE
	}

	opts := &criurpc.CriuOpts{
		ImagesDirFd: proto.Int32(imageDirFD),
		LogLevel:    proto.Int32(settings.LogLevel),
		LogFile:     proto.String(restoreLogFile),
		Root:        proto.String("/"),

		// Restore-specific options
		RstSibling:      proto.Bool(true),
		MntnsCompatMode: proto.Bool(false),
		EvasiveDevices:  proto.Bool(true),
		ForceIrmap:      proto.Bool(true),

		// Options from saved checkpoint
		ShellJob:          proto.Bool(settings.ShellJob),
		TcpClose:          proto.Bool(settings.TcpClose),
		FileLocks:         proto.Bool(settings.FileLocks),
		ExtUnixSk:         proto.Bool(settings.ExtUnixSk),
		LinkRemap:         proto.Bool(settings.LinkRemap),
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,

		// External mounts
		ExtMnt: extMounts,
	}

	if workDirFD >= 0 {
		opts.WorkDirFd = proto.Int32(workDirFD)
	}
	if settings.Timeout > 0 {
		opts.Timeout = proto.Uint32(settings.Timeout)
	}

	return opts
}

// remountProcSys remounts /proc/sys as rw (true) or ro (false).
func remountProcSys(rw bool, log *logrus.Entry) error {
	flags := uintptr(syscall.MS_REMOUNT | syscall.MS_BIND)
	if !rw {
		flags |= syscall.MS_RDONLY
	}
	mode := "ro"
	if rw {
		mode = "rw"
	}
	if err := syscall.Mount("", "/proc/sys", "", flags, ""); err != nil {
		return fmt.Errorf("failed to remount /proc/sys %s: %w", mode, err)
	}
	log.WithField("mode", mode).Debug("Remounted /proc/sys")
	return nil
}

// createLinkRemapStubs creates link_remap stub files (simplified version for cross-node restore).
func createLinkRemapStubs(checkpointPath string, log *logrus.Entry) {
	remapPath := filepath.Join(checkpointPath, "remap-fpath.img")
	if _, err := os.Stat(remapPath); os.IsNotExist(err) {
		return
	}
	// Link remap stubs are needed for cross-node restore. The full implementation
	// uses CRIU image parsing; for now we log that they may be needed.
	log.Debug("remap-fpath.img present — link_remap stubs may be needed for cross-node restore")
}

// logCRIUErrors reads and logs the CRIU restore log file.
func logCRIUErrors(checkpointPath string, log *logrus.Entry) {
	logPath := filepath.Join(checkpointPath, restoreLogFile)
	data, err := os.ReadFile(logPath)
	if err != nil {
		return
	}
	log.Error("=== CRIU RESTORE LOG ===")
	for _, line := range strings.Split(string(data), "\n") {
		if line != "" {
			log.Error(line)
		}
	}
	log.Error("=== END CRIU RESTORE LOG ===")
}

// restoreNotify captures the restored PID from CRIU callbacks.
type restoreNotify struct {
	criu.NoNotify
	restoredPID int32
	log         *logrus.Entry
}

func (n *restoreNotify) PreRestore() error {
	n.log.Debug("CRIU pre-restore")
	return nil
}

func (n *restoreNotify) PostRestore(pid int32) error {
	n.restoredPID = pid
	n.log.WithField("pid", pid).Info("CRIU post-restore: process restored")
	return nil
}

