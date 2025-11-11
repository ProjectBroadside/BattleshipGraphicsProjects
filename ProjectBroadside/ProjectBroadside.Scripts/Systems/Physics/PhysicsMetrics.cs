using Unity.Entities;

/// <summary>
/// A static class for collecting runtime performance metrics from the physics systems.
/// This allows for easy monitoring of key performance indicators without cluttering the individual systems.
/// </summary>
public static class PhysicsMetrics
{
    /// <summary>
    /// The time taken for the last buoyancy query, in milliseconds.
    /// </summary>
    public static float BuoyancyQueryTime { get; set; }

    /// <summary>
    /// The current number of active projectiles in the scene.
    /// </summary>
    public static int ActiveProjectileCount { get; set; }

    /// <summary>
    /// The overhead of the bridge systems, in milliseconds.
    /// This can be used to measure the performance impact of the DOTS-to-GameObject communication.
    /// </summary>
    public static float BridgeOverhead { get; set; }
}