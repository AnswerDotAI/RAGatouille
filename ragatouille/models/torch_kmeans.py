import torch
import time


def _train_kmeans(self, sample, shared_lists):
    if self.use_gpu:
        torch.cuda.empty_cache()
    centroids = compute_pytorch_kmeans(
        sample,
        self.config.dim,
        self.num_partitions,
        self.config.kmeans_niters,
        self.use_gpu,
    )
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    if self.use_gpu:
        centroids = centroids.half()
    else:
        centroids = centroids.float()
    return centroids


def compute_pytorch_kmeans(
    sample,
    dim,
    num_partitions,
    kmeans_niters,
    use_gpu,
    batch_size=512000,
    tol=1e-4,
    verbose=1,
):
    device = torch.device("cuda" if use_gpu else "cpu")
    sample = sample.to(device)
    total_size = sample.shape[0]

    # Initialize centroids randomly
    centroids = torch.randn(num_partitions, dim, dtype=sample.dtype, device=device)

    # Convert to half-precision if GPU is available
    if use_gpu:
        sample = sample.half()
        centroids = centroids.half()
    else:
        sample = sample.float()
        centroids = centroids.float()

    # Precompute the squared norms of data points
    sample_norms = torch.sum(sample.pow(2), dim=1, keepdim=True)

    start_time = time.time()
    for i in range(kmeans_niters):
        iter_time = time.time()

        # Shuffle the data points
        permutation = torch.randperm(total_size, device=device)
        sample = sample[permutation]
        sample_norms = sample_norms[permutation]

        if total_size <= batch_size:
            # Compute distances and assignments for the entire dataset
            distances = (
                sample_norms
                - 2 * torch.mm(sample, centroids.t())
                + torch.sum(centroids.pow(2), dim=1).unsqueeze(0)
            )
            assignments = torch.min(distances, dim=1)[1]

            # Update centroids by taking the mean of assigned data points
            for j in range(num_partitions):
                assigned_points = sample[assignments == j]
                if len(assigned_points) > 0:
                    centroids[j] = assigned_points.mean(dim=0)

            # Compute the error (sum of squared distances)
            error = torch.sum((sample - centroids[assignments]).pow(2))
        else:
            # Process the data points in batches
            error = 0.0
            for batch_start in range(0, total_size, batch_size):
                batch_end = min(batch_start + batch_size, total_size)
                batch = sample[batch_start:batch_end]
                batch_norms = sample_norms[batch_start:batch_end]

                # Compute distances and assignments for the batch
                distances = (
                    batch_norms
                    - 2 * torch.mm(batch, centroids.t())
                    + torch.sum(centroids.pow(2), dim=1).unsqueeze(0)
                )
                assignments = torch.min(distances, dim=1)[1]

                # Update centroids by taking the mean of assigned data points
                for j in range(num_partitions):
                    assigned_points = batch[assignments == j]
                    if len(assigned_points) > 0:
                        centroids[j] = (
                            centroids[j] * 0.9 + assigned_points.mean(dim=0) * 0.1
                        )

                # Accumulate the error for the batch
                error += torch.sum((batch - centroids[assignments]).pow(2))

        if verbose >= 2:
            print(
                f"Iteration: {i+1}, Error: {error.item():.4f}, Time: {time.time() - iter_time:.4f}s"
            )

        # Check for convergence (unlikely to early stop, but still useful!)
        if error <= tol:
            break

    if verbose >= 1:
        print(
            f"Used {i+1} iterations ({time.time() - start_time:.4f}s) to cluster {total_size} items into {num_partitions} clusters"
        )

    return centroids
