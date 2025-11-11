using System.Collections.Concurrent;
using Unity.Entities;

/// <summary>
/// A thread-safe queue for managing fire control requests from turrets.
/// This decouples the turret MonoBehaviours from the ECS firing systems.
/// </summary>
public static class FireRequestQueue
{
    /// <summary>
    /// Represents a request from a turret to fire a projectile.
    /// </summary>
    public struct FireRequest
    {
        /// <summary>
        /// The entity of the turret making the request.
        /// </summary>
        public Entity TurretEntity;
        /// <summary>
        /// The type of projectile to be fired.
        /// </summary>
        // public ProjectileType ProjectileType;
    }

    /// <summary>
    /// The internal concurrent queue storing fire requests.
    /// </summary>
    private static readonly ConcurrentQueue<FireRequest> RequestQueue = new ConcurrentQueue<FireRequest>();

    /// <summary>
    /// Adds a new fire request to the queue.
    /// </summary>
    /// <param name="request">The fire request to enqueue.</param>
    public static void Enqueue(FireRequest request)
    {
        RequestQueue.Enqueue(request);
    }

    /// <summary>
    /// Attempts to remove and return the fire request at the beginning of the queue.
    /// </summary>
    /// <param name="request">When this method returns, contains the dequeued request, if the operation was successful.</param>
    /// <returns>true if a request was successfully removed; otherwise, false.</returns>
    public static bool TryDequeue(out FireRequest request)
    {
        return RequestQueue.TryDequeue(out request);
    }

    /// <summary>
    /// Gets the current number of fire requests in the queue.
    /// </summary>
    public static int Count => RequestQueue.Count;
}