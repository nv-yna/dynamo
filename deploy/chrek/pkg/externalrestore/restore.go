// restore.go orchestrates external restore from the DaemonSet.
// All operations happen externally: rootfs replay via /host/proc/<PID>/root,
// CRIU restore via nsenter + criu-helper, CUDA restore via cuda-checkpoint.
package externalrestore

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/cuda"
)

const (
	// CRIUHelperBinary is the path to the criu-helper binary in the placeholder image.
	CRIUHelperBinary = "/usr/local/bin/criu-helper"

	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"
)

// RestorerConfig holds configuration for the external restore orchestrator.
type RestorerConfig struct {
	CheckpointBasePath string // Base path for checkpoint storage (PVC mount)
	CRIUHelperPath     string // Path to criu-helper binary (default: CRIUHelperBinary)
}

// Restorer orchestrates external restore operations from the DaemonSet.
type Restorer struct {
	cfg             RestorerConfig
	discoveryClient *checkpoint.DiscoveryClient
	log             *logrus.Entry
}

// NewRestorer creates a new external restore orchestrator.
func NewRestorer(cfg RestorerConfig, discoveryClient *checkpoint.DiscoveryClient) *Restorer {
	if cfg.CRIUHelperPath == "" {
		cfg.CRIUHelperPath = CRIUHelperBinary
	}
	return &Restorer{
		cfg:             cfg,
		discoveryClient: discoveryClient,
		log:             logrus.WithField("component", "restorer"),
	}
}

// Restore performs external restore for the given request.
func (r *Restorer) Restore(ctx context.Context, req RestoreAPIRequest) (*RestoreAPIResponse, error) {
	restoreStart := time.Now()
	r.log.WithFields(logrus.Fields{
		"checkpoint_id": req.CheckpointID,
		"pod":           req.PodName,
		"namespace":     req.PodNamespace,
		"container":     req.ContainerName,
	}).Info("=== Starting external restore ===")

	checkpointPath := filepath.Join(r.cfg.CheckpointBasePath, req.CheckpointID)

	// Load checkpoint manifest
	manifest, err := checkpoint.ReadCheckpointManifest(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	// Resolve the placeholder container to get its PID
	containerName := req.ContainerName
	if containerName == "" {
		containerName = "main"
	}

	placeholderPID, _, err := r.discoveryClient.ResolveContainerByPod(ctx, req.PodName, req.PodNamespace, containerName)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve placeholder container: %w", err)
	}
	r.log.WithField("pid", placeholderPID).Info("Resolved placeholder container")

	var completedSteps []string

	// Step 1: Apply rootfs diff into placeholder rootfs via /host/proc/<PID>/root
	targetRoot := fmt.Sprintf("%s/%d/root", checkpoint.HostProcPath, placeholderPID)
	if err := applyRootfsDiff(checkpointPath, targetRoot, r.log); err != nil {
		return nil, fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := applyDeletedFiles(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to apply deleted files")
	}
	completedSteps = append(completedSteps, "rootfs")

	// Step 2: Restore /dev/shm into placeholder
	if err := restoreDevShm(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to restore /dev/shm")
	}

	// Step 3: Create link_remap stubs in placeholder rootfs
	if err := createLinkRemapStubs(checkpointPath, targetRoot, r.log); err != nil {
		r.log.WithError(err).Warn("Failed to create link_remap stubs")
	}

	// Step 4: Execute nsenter + criu-helper
	restoredPID, err := r.executeCRIURestore(ctx, placeholderPID, checkpointPath, manifest)
	if err != nil {
		return nil, fmt.Errorf("CRIU restore failed: %w", err)
	}
	completedSteps = append(completedSteps, "criu")
	r.log.WithField("restored_pid", restoredPID).Info("CRIU restore completed")

	// Step 5: CUDA restore (if checkpoint has CUDA data)
	if manifest.ExternalRestore != nil && manifest.ExternalRestore.CUDA != nil {
		cudaStep := cuda.NewRestoreStep(r.log)
		if err := cudaStep.Execute(ctx, manifest, req.PodName, req.PodNamespace, containerName, restoredPID); err != nil {
			return nil, fmt.Errorf("CUDA restore failed: %w", err)
		}
		completedSteps = append(completedSteps, "cuda")
	}

	totalDuration := time.Since(restoreStart)
	r.log.WithFields(logrus.Fields{
		"total_duration": totalDuration,
		"restored_pid":   restoredPID,
		"steps":          completedSteps,
	}).Info("=== External restore completed ===")

	return &RestoreAPIResponse{
		Success:        true,
		RestoredPID:    restoredPID,
		CompletedSteps: completedSteps,
	}, nil
}

// executeCRIURestore runs criu-helper inside the placeholder's namespaces via nsenter.
func (r *Restorer) executeCRIURestore(ctx context.Context, placeholderPID int, checkpointPath string, manifest *checkpoint.CheckpointManifest) (int, error) {
	pidStr := strconv.Itoa(placeholderPID)

	// Build nsenter command: enter mount, network, PID, IPC namespaces
	args := []string{
		"-t", pidStr,
		"-m", "-n", "-p", "-i",
		"--", r.cfg.CRIUHelperPath,
		"--checkpoint-path", checkpointPath,
	}

	// Pass CRIU settings from manifest
	if manifest.CRIUDump.CRIU.WorkDir != "" {
		args = append(args, "--work-dir", manifest.CRIUDump.CRIU.WorkDir)
	}

	cmd := exec.CommandContext(ctx, "nsenter", args...)
	r.log.WithField("cmd", cmd.String()).Debug("Executing nsenter + criu-helper")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("nsenter + criu-helper failed: %w\noutput: %s", err, strings.TrimSpace(string(output)))
	}

	// Parse RESTORED_PID=<N> from stdout
	outputStr := string(output)
	for _, line := range strings.Split(outputStr, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "RESTORED_PID=") {
			pidStr := strings.TrimPrefix(line, "RESTORED_PID=")
			pid, err := strconv.Atoi(pidStr)
			if err != nil {
				return 0, fmt.Errorf("failed to parse RESTORED_PID from criu-helper output: %w", err)
			}
			return pid, nil
		}
	}

	return 0, fmt.Errorf("criu-helper did not output RESTORED_PID; output: %s", strings.TrimSpace(outputStr))
}

// applyRootfsDiff extracts rootfs-diff.tar into the target root.
func applyRootfsDiff(checkpointPath, targetRoot string, log *logrus.Entry) error {
	rootfsDiffPath := filepath.Join(checkpointPath, checkpoint.RootfsDiffFilename)
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.Debug("No rootfs-diff.tar, skipping")
		return nil
	}

	log.WithField("target", targetRoot).Info("Applying rootfs diff")
	cmd := exec.Command("tar", "--keep-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// tar exit codes 1-2 with --keep-old-files are non-fatal:
		// 1 = "file changed as we read it" or general warnings
		// 2 = "Cannot open: File exists" for read-only files
		// Both are expected when extracting over an existing rootfs.
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() <= 2 {
			log.WithField("output", string(output)).Debug("Rootfs diff applied (some files skipped)")
			return nil
		}
		return fmt.Errorf("tar extract failed: %w (output: %s)", err, string(output))
	}
	return nil
}

// applyDeletedFiles removes files marked as deleted in the checkpoint.
func applyDeletedFiles(checkpointPath, targetRoot string, log *logrus.Entry) error {
	deletedFilesPath := filepath.Join(checkpointPath, checkpoint.DeletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.WithError(err).WithField("path", target).Debug("Could not delete file")
			continue
		}
		count++
	}
	log.WithField("count", count).Info("Deleted files applied")
	return nil
}

// restoreDevShm restores /dev/shm files into the target root's /dev/shm.
func restoreDevShm(checkpointPath, targetRoot string, log *logrus.Entry) error {
	srcDir := filepath.Join(checkpointPath, checkpoint.DevShmDirName)
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to read dev-shm dir: %w", err)
	}

	destDir := filepath.Join(targetRoot, "dev", "shm")
	if err := os.MkdirAll(destDir, 0777); err != nil {
		return fmt.Errorf("failed to create target /dev/shm: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(srcDir, entry.Name())
		destPath := filepath.Join(destDir, entry.Name())

		info, err := entry.Info()
		if err != nil {
			continue
		}

		src, err := os.Open(srcPath)
		if err != nil {
			continue
		}

		mode := info.Mode()
		if mode == 0 {
			mode = 0666
		}
		dst, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			src.Close()
			continue
		}

		io.Copy(dst, src)
		src.Close()
		dst.Close()
	}

	log.WithField("count", len(entries)).Debug("Restored /dev/shm files")
	return nil
}

// createLinkRemapStubs creates link_remap stub files in the target rootfs.
// For external restore, stubs go into targetRoot instead of "/" since we operate
// from the DaemonSet's filesystem context.
func createLinkRemapStubs(checkpointPath, targetRoot string, log *logrus.Entry) error {
	// Check if remap-fpath.img exists
	remapPath := filepath.Join(checkpointPath, "remap-fpath.img")
	if _, err := os.Stat(remapPath); os.IsNotExist(err) {
		log.Debug("No remap-fpath.img, no link_remap stubs needed")
		return nil
	}

	// For now, delegate to the criu-helper which runs inside the container namespace
	// and has direct access to the container's filesystem.
	// The criu-helper will handle link_remap stubs as part of its restore flow.
	log.Debug("Link remap stubs will be handled by criu-helper inside container namespace")
	return nil
}
