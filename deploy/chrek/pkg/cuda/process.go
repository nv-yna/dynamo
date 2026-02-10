// process.go provides cgroup PID enumeration and CUDA process detection.
package cuda

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

const (
	// HostCgroupPath is the mount point for host cgroups in the DaemonSet.
	HostCgroupPath = "/sys/fs/cgroup"

	// NvidiaDriverLibPattern matches NVIDIA driver .so files that indicate CUDA usage.
	NvidiaDriverLibPattern = "libnvidia"
)

// GetCgroupPIDs returns all PIDs in the given container's cgroup.
// cgroupPath should be the container's cgroup path from /proc/<pid>/cgroup.
func GetCgroupPIDs(cgroupPath string) ([]int, error) {
	procsPath := filepath.Join(HostCgroupPath, cgroupPath, "cgroup.procs")
	data, err := os.ReadFile(procsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read cgroup.procs at %s: %w", procsPath, err)
	}

	var pids []int
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		pid, err := strconv.Atoi(line)
		if err != nil {
			continue
		}
		pids = append(pids, pid)
	}
	return pids, nil
}

// GetContainerCgroupPath reads the cgroup path for a given PID from /proc.
// For cgroup v2 (unified), this returns the path after "0::".
func GetContainerCgroupPath(pid int) (string, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", fmt.Errorf("failed to read cgroup for PID %d: %w", pid, err)
	}

	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		// cgroup v2 format: "0::<path>"
		if strings.HasPrefix(line, "0::") {
			return strings.TrimPrefix(line, "0::"), nil
		}
	}
	return "", fmt.Errorf("no cgroup v2 entry found for PID %d", pid)
}

// IsCUDAProcess checks if a process has CUDA libraries loaded by inspecting /proc/<pid>/maps.
func IsCUDAProcess(pid int) bool {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return false
	}
	return strings.Contains(string(data), NvidiaDriverLibPattern)
}

// FindCUDAPIDs returns the subset of pids that have CUDA libraries loaded.
func FindCUDAPIDs(pids []int, log *logrus.Entry) []int {
	var cudaPIDs []int
	for _, pid := range pids {
		if IsCUDAProcess(pid) {
			cudaPIDs = append(cudaPIDs, pid)
		}
	}
	log.WithFields(logrus.Fields{
		"total_pids": len(pids),
		"cuda_pids":  len(cudaPIDs),
	}).Debug("CUDA PID scan complete")
	return cudaPIDs
}
