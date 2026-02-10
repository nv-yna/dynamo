// Package cuda wraps the cuda-checkpoint binary for GPU state management.
// It provides lock, checkpoint, restore, and unlock operations used by the
// DaemonSet to manage CUDA state externally (without the CRIU CUDA plugin).
package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	// CudaCheckpointBinary is the path to the cuda-checkpoint binary.
	CudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"

	// DefaultTimeout is the default timeout for cuda-checkpoint operations.
	DefaultTimeout = 120 * time.Second
)

// Action represents a cuda-checkpoint operation.
type Action string

const (
	ActionLock       Action = "--toggle"
	ActionCheckpoint Action = "--checkpoint"
	ActionRestore    Action = "--restore"
	ActionUnlock     Action = "--toggle"
)

// RunCudaCheckpoint executes cuda-checkpoint for a single PID with the given action.
func RunCudaCheckpoint(ctx context.Context, pid int, action Action, log *logrus.Entry) error {
	args := []string{string(action), "--pid", fmt.Sprintf("%d", pid)}
	return runBinary(ctx, args, log)
}

// RunCudaCheckpointWithDeviceMap executes cuda-checkpoint restore with --device-map
// to remap GPUs from source UUIDs to target UUIDs.
func RunCudaCheckpointWithDeviceMap(ctx context.Context, pid int, deviceMap string, log *logrus.Entry) error {
	args := []string{string(ActionRestore), "--pid", fmt.Sprintf("%d", pid)}
	if deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	return runBinary(ctx, args, log)
}

// BuildDeviceMap creates a --device-map string mapping source GPU UUIDs to target GPU UUIDs.
// Format: "srcUUID0:dstUUID0,srcUUID1:dstUUID1,..."
func BuildDeviceMap(sourceUUIDs, targetUUIDs []string) (string, error) {
	if len(sourceUUIDs) != len(targetUUIDs) {
		return "", fmt.Errorf("GPU count mismatch: source has %d GPUs, target has %d",
			len(sourceUUIDs), len(targetUUIDs))
	}
	if len(sourceUUIDs) == 0 {
		return "", nil
	}

	pairs := make([]string, len(sourceUUIDs))
	for i := range sourceUUIDs {
		pairs[i] = sourceUUIDs[i] + ":" + targetUUIDs[i]
	}
	return strings.Join(pairs, ","), nil
}

func runBinary(ctx context.Context, args []string, log *logrus.Entry) error {
	timeoutCtx, cancel := context.WithTimeout(ctx, DefaultTimeout)
	defer cancel()

	cmd := exec.CommandContext(timeoutCtx, CudaCheckpointBinary, args...)
	log.WithField("args", args).Debug("Running cuda-checkpoint")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed: %w (output: %s)", args, err, strings.TrimSpace(string(output)))
	}
	return nil
}
