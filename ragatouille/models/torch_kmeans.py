import torch
from fast_pytorch_kmeans import KMeans


def _train_kmeans(self, sample, shared_lists):  # noqa: ARG001
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
    dim,  # noqa compatibility
    num_partitions,
    kmeans_niters,
    use_gpu,
    batch_size=16000,
    verbose=1,
    seed=123,
    max_points_per_centroid=256,
    min_points_per_centroid=10,
):
    device = torch.device("cuda" if use_gpu else "cpu")
    sample = sample.to(device)
    total_size = sample.shape[0]

    torch.manual_seed(seed)

    # Subsample the training set if too large
    if total_size > num_partitions * max_points_per_centroid:
        print("too many!")
        print("partitions:", num_partitions)
        print(total_size)
        perm = torch.randperm(total_size, device=device)[
            : num_partitions * max_points_per_centroid
        ]
        sample = sample[perm]
        total_size = sample.shape[0]
        print("reduced size:")
        print(total_size)
    elif total_size < num_partitions * min_points_per_centroid:
        if verbose:
            print(
                f"Warning: number of training points ({total_size}) is less than "
                f"the minimum recommended ({num_partitions * min_points_per_centroid})"
            )

    sample = sample.float()
    minibatch = None
    if num_partitions > 15000:
        minibatch = batch_size
    if num_partitions > 30000:
        minibatch = int(batch_size / 2)

    kmeans = KMeans(
        n_clusters=num_partitions,
        mode="euclidean",
        verbose=1,
        max_iter=kmeans_niters,
        minibatch=minibatch,
    )
    kmeans.fit(sample)
    return kmeans.centroids


""" Archived homebrew quick implementation to be revisited. """
# def compute_pytorch_kmeans(
#     sample,
#     dim,
#     num_partitions,
#     kmeans_niters,
#     use_gpu,
#     batch_size=8000,
#     tol=1e-4,
#     verbose=1,
#     seed=1234,
#     max_points_per_centroid=256,
#     min_points_per_centroid=10,
#     nredo=1,
# ):
#     device = torch.device("cuda" if use_gpu else "cpu")
#     sample = sample.to(device)
#     total_size = sample.shape[0]

#     torch.manual_seed(seed)

#     # Subsample the training set if too large
#     if total_size > num_partitions * max_points_per_centroid:
#         print("too many!")
#         print("partitions:", num_partitions)
#         print(total_size)
#         perm = torch.randperm(total_size, device=device)[
#             : num_partitions * max_points_per_centroid
#         ]
#         sample = sample[perm]
#         total_size = sample.shape[0]
#         print("reduced size:")
#         print(total_size)
#     elif total_size < num_partitions * min_points_per_centroid:
#         if verbose:
#             print(
#                 f"Warning: number of training points ({total_size}) is less than "
#                 f"the minimum recommended ({num_partitions * min_points_per_centroid})"
#             )

#     sample = sample.float()

#     best_obj = float("inf")
#     best_centroids = None

#     for redo in range(nredo):  # noqa
#         centroids = torch.randn(num_partitions, dim, dtype=sample.dtype, device=device)
#         centroids /= torch.norm(centroids, dim=1, keepdim=True)

#         with torch.no_grad():
#             for i in range(kmeans_niters):
#                 if verbose >= 1:
#                     print(f"KMEANS - Iteration {i+1} of {kmeans_niters}")
#                 start_time = time.time()
#                 obj = 0.0
#                 counts = torch.zeros(num_partitions, dtype=torch.long, device=device)

#                 # Process data in batches
#                 if verbose >= 1:
#                     _iterator = tqdm.tqdm(
#                         range(0, total_size, batch_size),
#                         total=math.ceil(total_size / batch_size),
#                         desc="Batches for iteration",
#                     )
#                 else:
#                     _iterator = range(0, total_size, batch_size)
#                 for batch_start in _iterator:
#                     batch_end = min(batch_start + batch_size, total_size)
#                     batch = sample[batch_start:batch_end]

#                     distances = torch.cdist(batch, centroids, p=2.0)
#                     batch_assignments = torch.min(distances, dim=1)[1]
#                     obj += torch.sum(
#                         distances[torch.arange(batch.size(0)), batch_assignments]
#                     )

#                     counts.index_add_(
#                         0, batch_assignments, torch.ones_like(batch_assignments)
#                     )

#                     for j in range(num_partitions):
#                         assigned_points = batch[batch_assignments == j]
#                         if len(assigned_points) > 0:
#                             centroids[j] += assigned_points.sum(dim=0)

#                 # Handle empty clusters by assigning them a random data point from the largest cluster
#                 empty_clusters = torch.where(counts == 0)[0]
#                 if empty_clusters.numel() > 0:
#                     for ec in empty_clusters:
#                         largest_cluster = torch.argmax(counts)
#                         idx = torch.randint(0, total_size, (1,), device=device)
#                         counts[largest_cluster] -= 1
#                         counts[ec] = 1
#                         centroids[ec] = sample[idx]

#                 centroids /= torch.norm(centroids, dim=1, keepdim=True)

#                 if verbose >= 2:
#                     print(
#                         f"Iteration: {i+1}, Objective: {obj.item():.4f}, Time: {time.time() - start_time:.4f}s"
#                     )

#                 # Check for convergence
#                 if obj < best_obj:
#                     best_obj = obj
#                     best_centroids = centroids.clone()
#                     if obj <= tol:
#                         break

#         torch.cuda.empty_cache()  # Move outside the inner loop

#     if verbose >= 1:
#         print(f"Best objective: {best_obj.item():.4f}")

#     print(best_centroids)

#     return best_centroids


""" Extremely slow implementation using voyager ANN below. CPU-only. Results ~= FAISS but would only be worth it if storing in int8, which might be for later."""
# from voyager import Index, Space

# def compute_pytorch_kmeans_via_voyager(
#     sample,
#     dim,
#     num_partitions,
#     kmeans_niters,
#     use_gpu,
#     batch_size=16000,
#     tol=1e-4,
#     verbose=3,
#     seed=1234,
#     max_points_per_centroid=256,
#     min_points_per_centroid=10,
#     nredo=1,
# ):
#     device = torch.device("cuda" if use_gpu else "cpu")
#     total_size = sample.shape[0]

#     # Set random seed for reproducibility
#     torch.manual_seed(seed)

#     # Convert to float32 for better performance
#     sample = sample.float()

#     best_obj = float("inf")
#     best_centroids = None

#     for redo in range(nredo):
#         # Initialize centroids randomly
#         centroids = torch.randn(num_partitions, dim, dtype=sample.dtype, device=device)
#         centroids = centroids / torch.norm(centroids, dim=1, keepdim=True)

#         # Build Voyager index if the number of data points exceeds 128,000
#         # use_index = total_size > 128000
#         use_index = True
#         if use_index:
#             index = Index(Space.Euclidean, num_dimensions=dim)
#             index.add_items(centroids.cpu().numpy())

#         for i in range(kmeans_niters):
#             start_time = time.time()
#             obj = 0.0
#             counts = torch.zeros(num_partitions, dtype=torch.long, device=device)

#             # Process data in batches
#             for batch_start in range(0, total_size, batch_size):
#                 batch_end = min(batch_start + batch_size, total_size)
#                 batch = sample[batch_start:batch_end].to(device)

#                 if use_index:
#                     # Search for nearest centroids using Voyager index
#                     batch_assignments, batch_distances = index.query(
#                         batch.cpu().numpy(), k=1, num_threads=-1
#                     )
#                     batch_assignments = (
#                         torch.from_numpy(batch_assignments.astype(np.int64))
#                         .squeeze()
#                         .to(device)
#                     )
#                     batch_assignments = batch_assignments.long()
#                     batch_distances = (
#                         torch.from_numpy(batch_distances.astype(np.float32))
#                         .squeeze()
#                         .to(device)
#                     )
#                 else:
#                     # Compute distances using memory-efficient operations
#                     distances = torch.cdist(batch, centroids, p=2.0)
#                     batch_assignments = torch.min(distances, dim=1)[1]
#                     batch_distances = distances[
#                         torch.arange(batch.size(0)), batch_assignments
#                     ]

#                 # Update objective and counts
#                 obj += torch.sum(batch_distances)
#                 counts += torch.bincount(batch_assignments, minlength=num_partitions)

#                 # Update centroids
#                 for j in range(num_partitions):
#                     assigned_points = batch[batch_assignments == j]
#                     if len(assigned_points) > 0:
#                         centroids[j] += assigned_points.sum(dim=0)

#                 # Clear the batch from memory
#                 del batch
#                 torch.cuda.empty_cache()

#             # Handle empty clusters
#             empty_clusters = torch.where(counts == 0)[0]
#             if empty_clusters.numel() > 0:
#                 for ec in empty_clusters:
#                     # Find the largest cluster
#                     largest_cluster = torch.argmax(counts)
#                     # Assign the empty cluster to a random data point from the largest cluster
#                     indexes = torch.where(counts == counts[largest_cluster])[0]
#                     if indexes.numel() > 0:
#                         idx = torch.randint(0, indexes.numel(), (1,), device=device)
#                         counts[largest_cluster] -= 1
#                         counts[ec] = 1
#                         centroids[ec] = sample[batch_start + indexes[idx].item()]
#             # Normalize centroids
#             centroids = centroids / torch.norm(centroids, dim=1, keepdim=True)

#             if use_index:
#                 # Update the Voyager index with the new centroids
#                 index = Index(Space.Euclidean, num_dimensions=dim)
#                 index.add_items(centroids.cpu().numpy())

#             if verbose >= 2:
#                 print(
#                     f"Iteration: {i+1}, Objective: {obj.item():.4f}, Time: {time.time() - start_time:.4f}s"
#                 )

#             # Check for convergence
#             if obj < best_obj:
#                 best_obj = obj
#                 best_centroids = centroids.clone()

#     if verbose >= 1:
#         print(f"Best objective: {best_obj.item():.4f}")

#     return best_centroids
