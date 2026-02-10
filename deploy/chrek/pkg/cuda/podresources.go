// podresources.go provides GPU UUID discovery via the kubelet PodResources gRPC API.
package cuda

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

const (
	// PodResourcesSocket is the default kubelet PodResources API socket path.
	// Mounted into the DaemonSet via hostPath.
	PodResourcesSocket = "/var/lib/kubelet/pod-resources/kubelet.sock"

	// NvidiaGPUResourceName is the Kubernetes extended resource name for NVIDIA GPUs.
	NvidiaGPUResourceName = "nvidia.com/gpu"
)

// GetPodGPUUUIDs queries the kubelet PodResources API for GPU UUIDs allocated to a pod.
// Returns nil (not an error) when the pod has no GPU resources.
func GetPodGPUUUIDs(ctx context.Context, podName, podNamespace, containerName string, log *logrus.Entry) ([]string, error) {
	dialCtx, dialCancel := context.WithTimeout(ctx, 10*time.Second)
	defer dialCancel()

	conn, err := grpc.DialContext(dialCtx, "unix://"+PodResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to PodResources API at %s: %w", PodResourcesSocket, err)
	}
	defer conn.Close()

	client := podresourcesv1.NewPodResourcesListerClient(conn)

	listCtx, listCancel := context.WithTimeout(ctx, 10*time.Second)
	defer listCancel()

	resp, err := client.List(listCtx, &podresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, fmt.Errorf("PodResources List failed: %w", err)
	}

	for _, pod := range resp.GetPodResources() {
		if pod.GetName() != podName || pod.GetNamespace() != podNamespace {
			continue
		}
		for _, container := range pod.GetContainers() {
			if containerName != "" && container.GetName() != containerName {
				continue
			}
			for _, device := range container.GetDevices() {
				if device.GetResourceName() != NvidiaGPUResourceName {
					continue
				}
				uuids := device.GetDeviceIds()
				log.WithFields(logrus.Fields{
					"pod":       podName,
					"namespace": podNamespace,
					"container": container.GetName(),
					"gpu_count": len(uuids),
				}).Debug("Found GPU UUIDs via PodResources API")
				return uuids, nil
			}
		}
	}

	return nil, nil
}
