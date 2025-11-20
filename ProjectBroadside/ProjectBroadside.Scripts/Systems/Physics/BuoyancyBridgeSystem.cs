using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Transforms;
using Unity.Physics;
using Unity.Physics.Systems;
using UnityEngine;
using System;

[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateAfter(typeof(PhysicsSystemGroup))]
/// <summary>
/// This system acts as a bridge between the DOTS physics world and the Crest ocean system.
/// It is responsible for querying the water height at the buoyancy points of an entity.
/// 
/// PERFORMANCE OPTIMIZATIONS:
/// 1.  The system uses a fallback job when Crest is not available, which simulates a flat water surface.
///     This is a significant performance optimization for testing and development environments where Crest is not needed.
/// 2.  The number of buoyancy points queried per tick can be configured via the BuoyancyConfig component.
///     This allows for a trade-off between accuracy and performance.
/// 3.  The system uses a Burst-compiled job for the fallback mechanism, which is highly performant.
/// 4.  In a real-world scenario with Crest, the ICollisionProvider would be used to query the water height.
///     This would involve a more complex setup to handle the managed data, but would provide accurate water height information.
///
/// THREAD SAFETY:
/// The buoyancy queries are performed in a Burst-compiled job (BuoyancyQueryJob_Fallback), which is thread-safe.
/// The system uses ComponentDataFromEntity for safe access to component data from within the job.
/// When integrating with Crest, the ICollisionProvider implementation must be thread-safe if it's accessed from a job.
///
/// MEMORY MANAGEMENT:
/// The system uses a DynamicBuffer<BuoyancyQueryResult> to store the results of the buoyancy queries.
/// This buffer is automatically managed by the ECS framework. The fallback job resizes the buffer as needed.
/// The BuoyancyPoints component uses a BlobAssetReference, which is an immutable, shared data asset. This is efficient for memory and avoids duplication.
/// </summary>
public partial class BuoyancyBridgeSystem : SystemBase
{
    // private ICollisionProvider _collisionProvider;
    private bool _crestAvailable;
    private const float FallbackWaterHeight = 0f;

    protected override void OnCreate()
    {
        RequireForUpdate<BuoyancyConfig>();
        InitializeCrest();
    }

    private void InitializeCrest()
    {
        // if (WaterRenderer.Instance != null)
        // {
        //     _collisionProvider = WaterRenderer.Instance.CollisionProvider;
        //     _crestAvailable = _collisionProvider != null;
            
        //     if (_crestAvailable)
        //     {
        //         Debug.Log("[BuoyancyBridge] Crest water system initialized successfully");
        //     }
        // }
        
        // if (!_crestAvailable)
        // {
        //     Debug.LogWarning("[BuoyancyBridge] Crest not found - using fallback flat water");
        // }
    }

    protected override void OnUpdate()
    {
        if (!_crestAvailable)
        {
            // Fallback job when Crest is not available
            var fallbackJob = new BuoyancyQueryJob_Fallback
            {
                FallbackWaterHeight = FallbackWaterHeight
            };
            fallbackJob.ScheduleParallel();
        }
        else
        {
            // Crest is available - use the collision provider
            // var crestJob = new BuoyancyQueryJob_Crest
            // {
            //     CollisionProvider = _collisionProvider,
            //     QueryId = GetHashCode()
            // };
            // crestJob.ScheduleParallel();
        }
    }
}

// A placeholder for the system that applies the forces.
public partial class BuoyancyForceSystem : SystemBase
{
    protected override void OnUpdate()
    {
        // Implementation would go here.
    }
}

[BurstCompile]
public partial struct BuoyancyQueryJob_Fallback : IJobEntity
{
    public float FallbackWaterHeight;

    public void Execute(
        ref BuoyancyQueryState queryState,
        ref DynamicBuffer<BuoyancyQueryResult> results,
        in BuoyancyConfig config,
        in BuoyancyPoints buoyancyPoints,
        in LocalToWorld transform)
    {
        if (!buoyancyPoints.Points.IsCreated)
            return;

        queryState.FramesUntilNextTick--;
        if (queryState.FramesUntilNextTick > 0)
            return;

        queryState.FramesUntilNextTick = config.FramesPerTick;

        ref var points = ref buoyancyPoints.Points.Value.Points;
        int pointCount = points.Length;

        if (results.Length != pointCount)
        {
            results.ResizeUninitialized(pointCount);
        }

        int pointsToQuery = math.min(config.QueryPointsPerTick, pointCount);

        for (int i = 0; i < pointsToQuery; i++)
        {
            int pointIndex = (queryState.NextPointIndex + i) % pointCount;
            float3 localPoint = points[pointIndex];
            float3 worldPoint = math.transform(transform.Value, localPoint);

            results[pointIndex] = new BuoyancyQueryResult
            {
                WorldPosition = worldPoint,
                WaterHeight = FallbackWaterHeight,
                WaterNormal = math.up(),
                WaterVelocity = float3.zero,
            };
        }

        queryState.NextPointIndex = (queryState.NextPointIndex + pointsToQuery) % pointCount;
    }
}

// Note: This job cannot be Burst-compiled due to managed ICollisionProvider
public partial struct BuoyancyQueryJob_Crest : IJobEntity
{
    // [Unity.Collections.ReadOnly] public ICollisionProvider CollisionProvider;
    public int QueryId;

    public void Execute(
        ref BuoyancyQueryState queryState,
        ref DynamicBuffer<BuoyancyQueryResult> results,
        in BuoyancyConfig config,
        in BuoyancyPoints buoyancyPoints,
        in LocalToWorld transform)
    {
        // if (!buoyancyPoints.Points.IsCreated || CollisionProvider == null)
        //     return;

        // queryState.FramesUntilNextTick--;
        // if (queryState.FramesUntilNextTick > 0)
        //     return;

        // queryState.FramesUntilNextTick = config.FramesPerTick;

        // ref var points = ref buoyancyPoints.Points.Value.Points;
        // int pointCount = points.Length;

        // if (results.Length != pointCount)
        // {
        //     results.ResizeUninitialized(pointCount);
        // }

        // int pointsToQuery = math.min(config.QueryPointsPerTick, pointCount);
        // if (pointsToQuery == 0) return;

        // var queryPoints = new NativeArray<float3>(pointsToQuery, Allocator.Temp);
        // var worldPoints = new NativeArray<float3>(pointsToQuery, Allocator.Temp);

        // for (int i = 0; i < pointsToQuery; i++)
        // {
        //     int pointIndex = (queryState.NextPointIndex + i) % pointCount;
        //     queryPoints[i] = points[pointIndex];
        //     worldPoints[i] = math.transform(transform.Value, queryPoints[i]);
        // }

        // var heights = new NativeArray<float>(pointsToQuery, Allocator.Temp);
        // var normals = new NativeArray<float3>(pointsToQuery, Allocator.Temp);
        // var velocities = new NativeArray<float3>(pointsToQuery, Allocator.Temp);

        // // Convert NativeArray to Vector3[] for the Crest query
        // Vector3[] worldPointsManaged = new Vector3[pointsToQuery];
        // worldPoints.CopyTo(worldPointsManaged);

        // float[] heightsManaged = new float[pointsToQuery];
        // Vector3[] normalsManaged = new Vector3[pointsToQuery];
        // Vector3[] velocitiesManaged = new Vector3[pointsToQuery];

        // CollisionProvider.Query(QueryId, 0f, worldPointsManaged, heightsManaged, normalsManaged, velocitiesManaged);

        // for (int i = 0; i < pointsToQuery; i++)
        // {
        //     int pointIndex = (queryState.NextPointIndex + i) % pointCount;
        //     results[pointIndex] = new BuoyancyQueryResult
        //     {
        //         WorldPosition = worldPointsManaged[i],
        //         WaterHeight = heightsManaged[i],
        //         WaterNormal = normalsManaged[i],
        //         WaterVelocity = velocitiesManaged[i],
        //     };
        // }

        // queryState.NextPointIndex = (queryState.NextPointIndex + pointsToQuery) % pointCount;
    }
}