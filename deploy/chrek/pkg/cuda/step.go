// step.go implements the CUDA restore step for external restore orchestration.
package cuda

import (
	"context"
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// RestoreStep performs CUDA restore + unlock after CRIU restore completes.
// It reads source GPU UUIDs from the checkpoint manifest, discovers target
// GPU UUIDs via PodResources API, builds a device map, and runs
// cuda-checkpoint restore+unlock for each CUDA PID.
type RestoreStep struct {
	log *logrus.Entry
}

// NewRestoreStep creates a new CUDA RestoreStep.
func NewRestoreStep(log *logrus.Entry) *RestoreStep {
	return &RestoreStep{log: log}
}

// Execute performs the CUDA restore for all CUDA PIDs in the checkpoint manifest.
// containerPID is the restored container's PID (used for cgroup enumeration).
func (s *RestoreStep) Execute(ctx context.Context, manifest *checkpoint.CheckpointManifest, podName, podNamespace, containerName string, containerPID int) error {
	if manifest.ExternalRestore == nil || manifest.ExternalRestore.CUDA == nil {
		s.log.Debug("No CUDA restore data in manifest, skipping")
		return nil
	}

	cudaData := manifest.ExternalRestore.CUDA
	if len(cudaData.PIDs) == 0 {
		s.log.Debug("No CUDA PIDs recorded, skipping")
		return nil
	}

	s.log.WithField("source_pids", cudaData.PIDs).Info("Starting CUDA restore")

	// Discover target GPU UUIDs via PodResources API
	targetUUIDs, err := GetPodGPUUUIDs(ctx, podName, podNamespace, containerName, s.log)
	if err != nil {
		return fmt.Errorf("failed to get target GPU UUIDs: %w", err)
	}

	// Build device map if source and target GPUs differ
	deviceMap := ""
	if len(cudaData.SourceGPUUUIDs) > 0 && len(targetUUIDs) > 0 {
		deviceMap, err = BuildDeviceMap(cudaData.SourceGPUUUIDs, targetUUIDs)
		if err != nil {
			return fmt.Errorf("failed to build device map: %w", err)
		}
		if deviceMap != "" {
			s.log.WithField("device_map", deviceMap).Info("GPU device mapping")
		}
	}

	// Find CUDA PIDs in the restored container's cgroup
	cgroupPath, err := GetContainerCgroupPath(containerPID)
	if err != nil {
		return fmt.Errorf("failed to get container cgroup path: %w", err)
	}

	cgroupPIDs, err := GetCgroupPIDs(cgroupPath)
	if err != nil {
		return fmt.Errorf("failed to enumerate cgroup PIDs: %w", err)
	}

	cudaPIDs := FindCUDAPIDs(cgroupPIDs, s.log)
	if len(cudaPIDs) == 0 {
		return fmt.Errorf("checkpoint manifest says %d CUDA PIDs but none found in restored cgroup (fail closed)", len(cudaData.PIDs))
	}

	// Restore + unlock each CUDA PID
	for _, pid := range cudaPIDs {
		if deviceMap != "" {
			if err := RunCudaCheckpointWithDeviceMap(ctx, pid, deviceMap, s.log); err != nil {
				return fmt.Errorf("cuda-checkpoint restore failed for PID %d: %w", pid, err)
			}
		} else {
			if err := RunCudaCheckpoint(ctx, pid, ActionRestore, s.log); err != nil {
				return fmt.Errorf("cuda-checkpoint restore failed for PID %d: %w", pid, err)
			}
		}
		s.log.WithField("pid", pid).Debug("CUDA restore completed for PID")

		// Unlock after restore
		if err := RunCudaCheckpoint(ctx, pid, ActionUnlock, s.log); err != nil {
			return fmt.Errorf("cuda-checkpoint unlock failed for PID %d: %w", pid, err)
		}
	}

	s.log.WithField("cuda_pids", len(cudaPIDs)).Info("CUDA restore completed")
	return nil
}
