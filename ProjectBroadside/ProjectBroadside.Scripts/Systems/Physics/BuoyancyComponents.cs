using Unity.Entities;
using Unity.Mathematics;

public struct BuoyancyQueryState : IComponentData
{
    public int NextPointIndex;
    public int FramesUntilNextTick;
}

public struct BuoyancyConfig : IComponentData
{
    public int QueryPointsPerTick;
    public int FramesPerTick;
    public float WaterDensity;
    public float BuoyancyMultiplier;
    public float HydrodynamicDrag;
}

public struct BuoyancyQueryResult : IBufferElementData
{
    public float3 WorldPosition;
    public float WaterHeight;
    public float3 WaterNormal;
    public float3 WaterVelocity;
}

public struct BuoyancyPoints : IComponentData
{
    public BlobAssetReference<BuoyancyPointsBlob> Points;
}

public struct BuoyancyPointsBlob
{
    public BlobArray<float3> Points;
}