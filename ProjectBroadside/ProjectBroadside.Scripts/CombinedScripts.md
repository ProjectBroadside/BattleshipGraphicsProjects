_ShipAndBuoyancy.cs
_Projectiles.cs
_Visuals.cs
BuoyancyBridgeSystem.cs
BuoyancyForceSystem.cs
ProjectileImpactSystem.cs
PhysicsCaptureSystem.cs
InterpolationSyncSystem.cs
ShipAuthoring.cs
ShipClassBlueprint.cs
BuoyancyProbeAsset.cs
PoolManager.cs
FireControlBridgeSystem.cs
FireRequestQueue.cs
ImpactEventBridgeSystem.cs
FireControlSystem.cs
EntityProxy.cs
ProxyManager.cs

// \=================================================================================================  
// FILE: ComponentDefinitions\_ShipAndBuoyancy.cs  
// \=================================================================================================

using Unity.Entities;  
using Unity.Mathematics;  
using Unity.Collections;  
using Unity.Transforms;  
using Unity.Physics;
using Unity.Physics.Systems;  
using Unity.Physics.Authoring;  
using UnityEngine;  
using WaveHarmonic.Crest;  
using System.Collections;  
using System.Collections.Generic;  
using System.Collections.Concurrent;

// \--- Ship identification and properties \---  
public struct ShipTag : IComponentData { }

public struct ShipProperties : IComponentData  
{  
    public float EnginePower;  
    public float RudderEffectiveness;  
    public float MaxSpeed;  
    public float TurnRate;  
}

// \--- Buoyancy components \---  
public struct BuoyancyConfig : IComponentData  
{  
    public int QueryPointsPerTick;  
    public int FramesPerTick;  
    public float WaterDensity;  
    public float BuoyancyMultiplier;  
    public float HydrodynamicDrag;  
}

public struct BuoyancyPoints : IComponentData  
{  
    public BlobAssetReference\<BuoyancyPointsBlob\> Points;  
}

public struct BuoyancyPointsBlob  
{  
    public BlobArray\<float3\> Points;  
}

public struct BuoyancyQueryState : IComponentData  
{  
    public int NextPointIndex;  
    public int FramesUntilNextTick;  
}

\[InternalBufferCapacity(64)\]  
public struct BuoyancyQueryResult : IBufferElementData  
{  
    public float3 WorldPosition;  
    public float WaterHeight;  
    public float3 WaterNormal;  
    public float3 WaterVelocity;  
    public bool IsValid;  
}

// \=================================================================================================  
// FILE: ComponentDefinitions\_Projectiles.cs  
// \=================================================================================================

// \--- Projectile tags and properties \---  
public struct ActiveProjectileTag : IComponentData { }  
public struct PooledProjectileTag : IComponentData { }

public struct BallisticProperties : IComponentData  
{  
    public float DragCoefficient;  
    public float CrossSectionalArea;  
    public float Mass;  
}

public struct ProjectileData : IComponentData  
{  
    public Entity FiredBy;  
    public float3 InitialVelocity;  
    public float TimeAlive;  
    public float MaxLifetime;  
}

// \--- Impact event data \---  
public struct ImpactEvent : IComponentData  
{  
    public Entity Target;  
    public float3 ImpactPosition;  
    public float3 ImpactNormal;  
    public float3 ProjectileVelocity;  
    public float ProjectileMass;  
    public float ImpactEnergy;  
}

// \=================================================================================================  
// FILE: ComponentDefinitions\_Visuals.cs  
// \=================================================================================================

public struct InterpolatedTransform : IComponentData  
{  
    public RigidTransform PreviousTransform;  
    public RigidTransform CurrentTransform;  
    public float LastCaptureTime;  
}

public struct VisualProxy : IComponentData  
{  
    public Entity ProxyEntity;  
}

// \=================================================================================================  
// FILE: BuoyancyBridgeSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))\]  
\[UpdateBefore(typeof(BuoyancyForceSystem))\]  
public partial class BuoyancyBridgeSystem : SystemBase  
{  
    private ICollisionProvider \_collisionProvider;  
    private bool \_crestAvailable;  
    private const float FallbackWaterHeight \= 0f;

    protected override void OnCreate()  
    {  
        RequireForUpdate\<BuoyancyConfig\>();  
        InitializeCrest();  
    }

    private void InitializeCrest()  
    {  
        if (WaterRenderer.Instance \!= null)  
        {  
            \_collisionProvider \= WaterRenderer.Instance.CollisionProvider;  
            \_crestAvailable \= \_collisionProvider \!= null;  
              
            if (\_crestAvailable)  
            {  
                Debug.Log("\[BuoyancyBridge\] Crest water system initialized successfully");  
            }  
        }  
          
        if (\!\_crestAvailable)  
        {  
            Debug.LogWarning("\[BuoyancyBridge\] Crest not found \- using fallback flat water");  
        }  
    }

    protected override void OnUpdate()  
    {  
        var crestAvailable \= \_crestAvailable;  
        var collisionProvider \= \_collisionProvider;  
        var queryId \= GetHashCode();

        Entities  
            .WithName("UpdateBuoyancyQueries")  
            .ForEach((  
                Entity entity,  
                ref BuoyancyQueryState queryState,  
                ref DynamicBuffer\<BuoyancyQueryResult\> results,  
                in BuoyancyConfig config,  
                in BuoyancyPoints buoyancyPoints,  
                in LocalToWorld transform) \=\>  
            {  
                if (\!buoyancyPoints.Points.IsCreated)  
                    return;

                // Check if it's time to update  
                queryState.FramesUntilNextTick--;  
                if (queryState.FramesUntilNextTick \> 0\)  
                    return;

                queryState.FramesUntilNextTick \= config.FramesPerTick;

                ref var points \= ref buoyancyPoints.Points.Value.Points;  
                int pointCount \= points.Length;

                // Ensure buffer is correct size  
                if (results.Length \!= pointCount)  
                {  
                    results.ResizeUninitialized(pointCount);  
                }

                // Query points for this frame  
                int pointsToQuery \= math.min(config.QueryPointsPerTick, pointCount);  
                  
                for (int i \= 0; i \< pointsToQuery; i++)  
                {  
                    int pointIndex \= (queryState.NextPointIndex \+ i) % pointCount;  
                    float3 localPoint \= points\[pointIndex\];  
                    float3 worldPoint \= math.transform(transform.Value, localPoint);

                    if (crestAvailable)  
                    {  
                        var collisionStatus \= collisionProvider.Query(  
                            queryId,  
                            worldPoint.y,  
                            worldPoint,  
                            out var collisionResult);

                        results\[pointIndex\] \= new BuoyancyQueryResult  
                        {  
                            WorldPosition \= worldPoint,  
                            WaterHeight \= collisionResult.position.y,  
                            WaterNormal \= collisionResult.normal,  
                            WaterVelocity \= collisionResult.waterSurfaceVel,  
                            IsValid \= collisionStatus  
                        };  
                    }  
                    else  
                    {  
                        // Fallback: flat water  
                        results\[pointIndex\] \= new BuoyancyQueryResult  
                        {  
                            WorldPosition \= worldPoint,  
                            WaterHeight \= FallbackWaterHeight,  
                            WaterNormal \= new float3(0, 1, 0),  
                            WaterVelocity \= float3.zero,  
                            IsValid \= true  
                        };  
                    }  
                }

                queryState.NextPointIndex \= (queryState.NextPointIndex \+ pointsToQuery) % pointCount;  
            })  
            .WithoutBurst()  
            .Run();  
    }  
}

// \=================================================================================================  
// FILE: BuoyancyForceSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))\]  
\[UpdateAfter(typeof(BuoyancyBridgeSystem))\]  
\[UpdateBefore(typeof(PhysicsSystemGroup))\]  
public partial struct BuoyancyForceSystem : ISystem  
{  
    \[Unity.Burst.BurstCompile\]  
    public void OnUpdate(ref SystemState state)  
    {  
        var deltaTime \= SystemAPI.Time.DeltaTime;  
          
        new BuoyancyForceJob  
        {  
            DeltaTime \= deltaTime,  
            Gravity \= new float3(0, \-9.81f, 0\)  
        }.ScheduleParallel();  
    }  
}

\[Unity.Burst.BurstCompile\]  
public partial struct BuoyancyForceJob : IJobEntity  
{  
    public float DeltaTime;  
    public float3 Gravity;

    public void Execute(  
        ref PhysicsVelocity velocity,  
        ref PhysicsMass mass,  
        in DynamicBuffer\<BuoyancyQueryResult\> queryResults,  
        in BuoyancyConfig config,  
        in LocalToWorld transform)  
    {  
        if (queryResults.IsEmpty)  
            return;

        float3 totalBuoyancyForce \= float3.zero;  
        float3 totalDragForce \= float3.zero;  
        int validPoints \= 0;

        // Calculate buoyancy forces  
        for (int i \= 0; i \< queryResults.Length; i++)  
        {  
            var result \= queryResults\[i\];  
            if (\!result.IsValid)  
                continue;

            float submersion \= result.WaterHeight \- result.WorldPosition.y;  
            if (submersion \> 0\)  
            {  
                // Archimedes' principle: F \= ρ \* V \* g  
                float forceMagnitude \= config.WaterDensity \* submersion \* \-Gravity.y;  
                float3 buoyancyForce \= result.WaterNormal \* forceMagnitude \* config.BuoyancyMultiplier;  
                  
                totalBuoyancyForce \+= buoyancyForce;

                // Hydrodynamic drag  
                float3 relativeVelocity \= velocity.Linear \- result.WaterVelocity;  
                float3 dragForce \= \-config.HydrodynamicDrag \* relativeVelocity \* submersion;  
                totalDragForce \+= dragForce;

                validPoints++;  
            }  
        }

        if (validPoints \> 0\)  
        {  
            // Average the forces  
            totalBuoyancyForce /= validPoints;  
            totalDragForce /= validPoints;

            // Apply forces  
            float3 totalForce \= totalBuoyancyForce \+ totalDragForce;  
            velocity.Linear \+= totalForce \* mass.InverseMass \* DeltaTime;

            // Apply angular damping to prevent excessive rotation  
            velocity.Angular \*= math.lerp(1f, 0.95f, DeltaTime);  
        }  
    }  
}

// \=================================================================================================  
// FILE: ProjectileImpactSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))\]  
\[UpdateAfter(typeof(PhysicsSystemGroup))\]  
public partial struct ProjectileImpactSystem : ISystem  
{  
    private ComponentLookup\<ShipTag\> \_shipTagLookup;  
    private ComponentLookup\<ProjectileData\> \_projectileDataLookup;  
    private ComponentLookup\<PhysicsMass\> \_massLookup;

    \[Unity.Burst.BurstCompile\]  
    public void OnCreate(ref SystemState state)  
    {  
        \_shipTagLookup \= state.GetComponentLookup\<ShipTag\>(true);  
        \_projectileDataLookup \= state.GetComponentLookup\<ProjectileData\>(true);  
        \_massLookup \= state.GetComponentLookup\<PhysicsMass\>(true);  
    }

    \[Unity.Burst.BurstCompile\]  
    public void OnUpdate(ref SystemState state)  
    {  
        \_shipTagLookup.Update(ref state);  
        \_projectileDataLookup.Update(ref state);  
        \_massLookup.Update(ref state);

        var ecb \= new EntityCommandBuffer(Allocator.TempJob);  
          
        state.Dependency \= new ProjectileImpactJob  
        {  
            ShipTagLookup \= \_shipTagLookup,  
            ProjectileDataLookup \= \_projectileDataLookup,  
            MassLookup \= \_massLookup,  
            ECB \= ecb.AsParallelWriter()  
        }.Schedule(SystemAPI.GetSingleton\<SimulationSingleton\>(), state.Dependency);

        state.Dependency.Complete();  
        ecb.Playback(state.EntityManager);  
        ecb.Dispose();  
    }  
}

\[Unity.Burst.BurstCompile\]  
struct ProjectileImpactJob : ICollisionEventsJob  
{  
    \[ReadOnly\] public ComponentLookup\<ShipTag\> ShipTagLookup;  
    \[ReadOnly\] public ComponentLookup\<ProjectileData\> ProjectileDataLookup;  
    \[ReadOnly\] public ComponentLookup\<PhysicsMass\> MassLookup;  
    public EntityCommandBuffer.ParallelWriter ECB;

    public void Execute(CollisionEvent collisionEvent)  
    {  
        Entity projectileEntity \= Entity.Null;  
        Entity targetEntity \= Entity.Null;

        // Determine which entity is the projectile  
        bool entityAIsProjectile \= ProjectileDataLookup.HasComponent(collisionEvent.EntityA);  
        bool entityBIsProjectile \= ProjectileDataLookup.HasComponent(collisionEvent.EntityB);

        if (entityAIsProjectile)  
        {  
            projectileEntity \= collisionEvent.EntityA;  
            targetEntity \= collisionEvent.EntityB;  
        }  
        else if (entityBIsProjectile)  
        {  
            projectileEntity \= collisionEvent.EntityB;  
            targetEntity \= collisionEvent.EntityA;  
        }  
        else  
        {  
            return; // Neither entity is a projectile  
        }

        // Check if target is a ship  
        if (\!ShipTagLookup.HasComponent(targetEntity))  
            return;

        // Get projectile data  
        var projectileData \= ProjectileDataLookup\[projectileEntity\];  
        var projectileMass \= MassLookup\[projectileEntity\];

        // Calculate impact data  
        float3 impactVelocity \= collisionEvent.Normal \* math.length(projectileData.InitialVelocity);  
        float impactEnergy \= 0.5f \* (1f / projectileMass.InverseMass) \* math.lengthsq(impactVelocity);

        // Create impact event on the target  
        ECB.AddComponent(0, targetEntity, new ImpactEvent  
        {  
            Target \= targetEntity,  
            ImpactPosition \= collisionEvent.CalculateDetails(ref collisionEvent).AverageContactPointPosition,  
            ImpactNormal \= collisionEvent.Normal,  
            ProjectileVelocity \= impactVelocity,  
            ProjectileMass \= 1f / projectileMass.InverseMass,  
            ImpactEnergy \= impactEnergy  
        });

        // Disable the projectile  
        ECB.RemoveComponent\<ActiveProjectileTag\>(0, projectileEntity);  
        ECB.AddComponent\<PooledProjectileTag\>(0, projectileEntity);  
        ECB.AddComponent\<PhysicsExclude\>(0, projectileEntity);  
    }  
}

// \=================================================================================================  
// FILE: PhysicsCaptureSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(SimulationSystemGroup))\]  
\[UpdateAfter(typeof(TransformSystemGroup))\]  
public partial struct PhysicsCaptureSystem : ISystem  
{  
    \[Unity.Burst.BurstCompile\]  
    public void OnUpdate(ref SystemState state)  
    {  
        var currentTime \= SystemAPI.Time.ElapsedTime;  
          
        new PhysicsCaptureJob  
        {  
            CurrentTime \= (float)currentTime  
        }.ScheduleParallel();  
    }  
}

\[Unity.Burst.BurstCompile\]  
public partial struct PhysicsCaptureJob : IJobEntity  
{  
    public float CurrentTime;

    public void Execute(ref InterpolatedTransform interpolated, in LocalToWorld transform)  
    {  
        // Store previous transform  
        interpolated.PreviousTransform \= interpolated.CurrentTransform;  
          
        // Update current transform  
        interpolated.CurrentTransform \= new RigidTransform(transform.Rotation, transform.Position);  
          
        // Update capture time  
        interpolated.LastCaptureTime \= CurrentTime;  
    }  
}

// \=================================================================================================  
// FILE: InterpolationSyncSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(PresentationSystemGroup))\]  
public partial class InterpolationSyncSystem : SystemBase  
{  
    protected override void OnCreate()  
    {  
        RequireForUpdate\<InterpolatedTransform\>();  
    }

    protected override void OnUpdate()  
    {  
        float interpolationFactor \= SystemAPI.Time.DeltaTime / Time.fixedDeltaTime;  
        interpolationFactor \= math.saturate(interpolationFactor);

        Entities  
            .WithName("UpdateVisualProxies")  
            .WithoutBurst()  
            .ForEach((Entity entity, in InterpolatedTransform interpolated) \=\>  
            {  
                if (ProxyManager.Instance \!= null &&   
                    ProxyManager.Instance.TryGetProxy(entity, out EntityProxy proxy))  
                {  
                    // Interpolate rotation using slerp  
                    quaternion interpolatedRotation \= math.slerp(  
                        interpolated.PreviousTransform.rot,  
                        interpolated.CurrentTransform.rot,  
                        interpolationFactor);

                    // Interpolate position using lerp  
                    float3 interpolatedPosition \= math.lerp(  
                        interpolated.PreviousTransform.pos,  
                        interpolated.CurrentTransform.pos,  
                        interpolationFactor);

                    // Apply to GameObject  
                    proxy.transform.SetPositionAndRotation(  
                        interpolatedPosition,  
                        interpolatedRotation);  
                }  
            }).Run();  
    }  
}

// \=================================================================================================  
// FILE: ShipAuthoring.cs  
// \=================================================================================================

\[DisallowMultipleComponent\]  
public class ShipAuthoring : MonoBehaviour  
{  
    \[Header("Ship Configuration")\]  
    public ShipClassBlueprint shipBlueprint;  
      
    \[Header("Physics")\]  
    public PhysicsShapeAuthoring hullCollider;  
      
    \[Header("Buoyancy")\]  
    public BuoyancyProbeAsset buoyancyProbes;

    class Baker : Baker\<ShipAuthoring\>  
    {  
        public override void Bake(ShipAuthoring authoring)  
        {  
            if (\!ValidateAuthoring(authoring))  
                return;

            var entity \= GetEntity(TransformUsageFlags.Dynamic);  
              
            // Core components  
            AddComponent\<ShipTag\>(entity);  
              
            // Ship properties  
            AddComponent(entity, new ShipProperties  
            {  
                EnginePower \= authoring.shipBlueprint.enginePower,  
                RudderEffectiveness \= authoring.shipBlueprint.rudderEffectiveness,  
                MaxSpeed \= authoring.shipBlueprint.maxSpeed,  
                TurnRate \= authoring.shipBlueprint.turnRate  
            });

            // Physics  
            var colliderBlob \= authoring.hullCollider.GetBakedConvexProperties();  
            AddComponent(entity, new PhysicsCollider { Value \= colliderBlob });  
              
            var mass \= PhysicsMass.CreateDynamic(  
                authoring.hullCollider.GetMassProperties(),  
                authoring.shipBlueprint.massInKg);  
            AddComponent(entity, mass);  
              
            AddComponent\<PhysicsVelocity\>(entity);  
            AddComponent\<PhysicsGravityFactor\>(entity, new PhysicsGravityFactor { Value \= 0f });

            // Buoyancy  
            AddComponent(entity, new BuoyancyConfig  
            {  
                QueryPointsPerTick \= authoring.shipBlueprint.queryPointsPerTick,  
                FramesPerTick \= authoring.shipBlueprint.framesPerTick,  
                WaterDensity \= 1025f, // Sea water density  
                BuoyancyMultiplier \= authoring.shipBlueprint.buoyancyMultiplier,  
                HydrodynamicDrag \= authoring.shipBlueprint.hydrodynamicDrag  
            });

            var probeReference \= authoring.buoyancyProbes.GetBlobReference();  
            AddComponent(entity, new BuoyancyPoints { Points \= probeReference });  
              
            AddComponent\<BuoyancyQueryState\>(entity);  
            AddBuffer\<BuoyancyQueryResult\>(entity);

            // Visual interpolation  
            AddComponent\<InterpolatedTransform\>(entity);  
        }

        private bool ValidateAuthoring(ShipAuthoring authoring)  
        {  
            if (authoring.shipBlueprint \== null)  
            {  
                Debug.LogError($"Ship blueprint missing on {authoring.name}", authoring);  
                return false;  
            }

            if (authoring.hullCollider \== null)  
            {  
                Debug.LogError($"Hull collider missing on {authoring.name}", authoring);  
                return false;  
            }

            if (authoring.buoyancyProbes \== null)  
            {  
                Debug.LogError($"Buoyancy probes missing on {authoring.name}", authoring);  
                return false;  
            }

            return true;  
        }  
    }  
}

// \=================================================================================================  
// FILE: ShipClassBlueprint.cs  
// \=================================================================================================

\[CreateAssetMenu(fileName \= "ShipClass\_", menuName \= "Naval Combat/Ship Class Blueprint")\]  
public class ShipClassBlueprint : ScriptableObject  
{  
    \[Header("Identity")\]  
    public string className \= "Destroyer";  
    public string displayName \= "HMS Example";  
      
    \[Header("Physics Properties")\]  
    \[Tooltip("Mass in kilograms")\]  
    public float massInKg \= 8000000f; // 8,000 tons  
      
    \[Header("Movement")\]  
    public float enginePower \= 75000f; // 75 MW  
    public float rudderEffectiveness \= 45f;  
    public float maxSpeed \= 30f; // knots  
    public float turnRate \= 2.5f; // degrees per second at full rudder  
      
    \[Header("Buoyancy Simulation")\]  
    \[Tooltip("Number of buoyancy points to query each physics tick")\]  
    public int queryPointsPerTick \= 16;  
      
    \[Tooltip("Number of frames between full buoyancy updates")\]  
    public int framesPerTick \= 1;  
      
    \[Tooltip("Multiplier for buoyancy force")\]  
    public float buoyancyMultiplier \= 1.2f;  
      
    \[Tooltip("Hydrodynamic drag coefficient")\]  
    public float hydrodynamicDrag \= 0.98f;  
      
    \[Header("Combat Properties")\]  
    public float maxHealth \= 1000f;  
    public float armorThickness \= 200f; // mm  
}

// \=================================================================================================  
// FILE: BuoyancyProbeAsset.cs  
// \=================================================================================================

\[CreateAssetMenu(fileName \= "BuoyancyProbes\_", menuName \= "Naval Combat/Buoyancy Probe Asset")\]  
public class BuoyancyProbeAsset : ScriptableObject  
{  
    \[HideInInspector\]  
    public byte\[\] serializedBlobData;  
      
    private BlobAssetReference\<BuoyancyPointsBlob\> \_cachedBlob;  
      
    public BlobAssetReference\<BuoyancyPointsBlob\> GetBlobReference()  
    {  
        if (\_cachedBlob.IsCreated)  
            return \_cachedBlob;  
              
        if (serializedBlobData \== null || serializedBlobData.Length \== 0\)  
        {  
            Debug.LogError($"No buoyancy probe data found in {name}");  
            return BlobAssetReference\<BuoyancyPointsBlob\>.Null;  
        }  
          
        // Deserialize the blob  
        \_cachedBlob \= DeserializeBlob(serializedBlobData);  
        return \_cachedBlob;  
    }  
      
    public void SetProbeData(float3\[\] probePoints)  
    {  
        // Create blob  
        var builder \= new BlobBuilder(Allocator.Temp);  
        ref var root \= ref builder.ConstructRoot\<BuoyancyPointsBlob\>();  
          
        var arrayBuilder \= builder.Allocate(ref root.Points, probePoints.Length);  
        for (int i \= 0; i \< probePoints.Length; i++)  
        {  
            arrayBuilder\[i\] \= probePoints\[i\];  
        }  
          
        \_cachedBlob \= builder.CreateBlobAssetReference\<BuoyancyPointsBlob\>(Allocator.Persistent);  
        builder.Dispose();  
          
        // Serialize for storage  
        serializedBlobData \= SerializeBlob(\_cachedBlob);  
          
\#if UNITY\_EDITOR  
        UnityEditor.EditorUtility.SetDirty(this);  
\#endif  
    }  
      
    private byte\[\] SerializeBlob(BlobAssetReference\<BuoyancyPointsBlob\> blob)  
    {  
        using (var writer \= new Unity.Entities.Serialization.MemoryBinaryWriter())  
        {  
            writer.Write(blob);  
            return writer.Data.ToArray();  
        }  
    }  
      
    private BlobAssetReference\<BuoyancyPointsBlob\> DeserializeBlob(byte\[\] data)  
    {  
        unsafe  
        {  
            fixed (byte\* ptr \= data)  
            {  
                var reader \= new Unity.Entities.Serialization.MemoryBinaryReader(ptr, data.Length);  
                return reader.Read\<BuoyancyPointsBlob\>();  
            }  
        }  
    }  
      
    private void OnDisable()  
    {  
        if (\_cachedBlob.IsCreated)  
        {  
            \_cachedBlob.Dispose();  
        }  
    }  
}

// \=================================================================================================  
// FILE: PoolManager.cs  
// \=================================================================================================

public class PoolManager : MonoBehaviour  
{  
    \[Header("Projectile Pool Configuration")\]  
    public GameObject projectilePrefab;  
    public int poolSize \= 5000;  
    public int batchSize \= 100;  
      
    private Entity \_projectilePrototype;  
    private EntityManager \_entityManager;  
    private bool \_isInitialized;

    private void Start()  
    {  
        Initialize();  
    }

    private void Initialize()  
    {  
        var world \= World.DefaultGameObjectInjectionWorld;  
        if (world \== null)  
        {  
            Debug.LogError("No ECS world found\!");  
            return;  
        }

        \_entityManager \= world.EntityManager;  
          
        // Convert prefab to entity prototype  
        var conversionSettings \= GameObjectConversionSettings.FromWorld(world, null);  
        \_projectilePrototype \= GameObjectConversionUtility.ConvertGameObjectHierarchy(  
            projectilePrefab,   
            conversionSettings);  
          
        StartCoroutine(CreatePoolAsync());  
    }

    private IEnumerator CreatePoolAsync()  
    {  
        Debug.Log($"\[PoolManager\] Starting projectile pool creation (size: {poolSize})");  
          
        int created \= 0;  
        var archetype \= \_entityManager.CreateArchetype(  
            typeof(PooledProjectileTag),  
            typeof(PhysicsExclude),  
            typeof(LocalTransform),  
            typeof(LocalToWorld));

        while (created \< poolSize)  
        {  
            using (var ecb \= new EntityCommandBuffer(Allocator.TempJob))  
            {  
                int batchEnd \= Mathf.Min(created \+ batchSize, poolSize);  
                  
                for (int i \= created; i \< batchEnd; i++)  
                {  
                    var entity \= ecb.Instantiate(\_projectilePrototype);  
                    ecb.AddComponent\<PooledProjectileTag\>(entity);  
                    ecb.AddComponent\<PhysicsExclude\>(entity);  
                }  
                  
                ecb.Playback(\_entityManager);  
                created \= batchEnd;  
            }  
              
            // Yield to prevent frame drops  
            yield return null;  
        }  
          
        \_isInitialized \= true;  
        Debug.Log($"\[PoolManager\] Pool creation complete ({created} projectiles)");  
    }

    public bool IsReady \=\> \_isInitialized;  
}

// \=================================================================================================  
// FILE: FireControlBridgeSystem.cs  
// \=================================================================================================

/// \<summary\>  
/// Bridges fire control requests from MonoBehaviour FireControlSystem to DOTS projectiles  
/// \</summary\>  
\[UpdateInGroup(typeof(SimulationSystemGroup))\]  
public partial class FireControlBridgeSystem : SystemBase  
{  
    private EntityCommandBufferSystem \_ecbSystem;  
    private EntityQuery \_pooledProjectileQuery;  
      
    protected override void OnCreate()  
    {  
        \_ecbSystem \= World.GetOrCreateSystemManaged\<BeginSimulationEntityCommandBufferSystem\>();  
        \_pooledProjectileQuery \= GetEntityQuery(  
            ComponentType.ReadOnly\<PooledProjectileTag\>(),  
            ComponentType.Exclude\<ActiveProjectileTag\>());  
    }  
      
    protected override void OnUpdate()  
    {  
        var ecb \= \_ecbSystem.CreateCommandBuffer();  
        var availableProjectiles \= \_pooledProjectileQuery.ToEntityArray(Allocator.TempJob);  
          
        // Process fire requests from MonoBehaviour queue  
        while (FireRequestQueue.Instance.TryDequeue(out var request))  
        {  
            if (availableProjectiles.Length \> 0\)  
            {  
                var projectile \= availableProjectiles\[availableProjectiles.Length \- 1\];  
                availableProjectiles.RemoveAtSwapBack(availableProjectiles.Length \- 1);  
                  
                // Activate projectile with fire control data  
                ecb.RemoveComponent\<PooledProjectileTag\>(projectile);  
                ecb.AddComponent\<ActiveProjectileTag\>(projectile);  
                ecb.RemoveComponent\<PhysicsExclude\>(projectile);  
                  
                ecb.SetComponent(projectile, new ProjectileData  
                {  
                    FiredBy \= request.FiringShip,  
                    InitialVelocity \= request.Velocity,  
                    TimeAlive \= 0f,  
                    MaxLifetime \= request.MaxFlightTime  
                });  
                  
                ecb.SetComponent(projectile, request.Transform);  
            }  
        }  
          
        availableProjectiles.Dispose();  
    }  
}

// \=================================================================================================  
// FILE: FireRequestQueue.cs  
// \=================================================================================================

public class FireRequestQueue  
{  
    public static FireRequestQueue Instance { get; } \= new FireRequestQueue();  
      
    private readonly ConcurrentQueue\<FireRequest\> \_requests \= new ConcurrentQueue\<FireRequest\>();  
      
    public void Enqueue(FireRequest request) \=\> \_requests.Enqueue(request);  
    public bool TryDequeue(out FireRequest request) \=\> \_requests.TryDequeue(out request);  
}

public struct FireRequest  
{  
    public Entity FiringShip;  
    public LocalTransform Transform;  
    public float3 Velocity;  
    public float MaxFlightTime;  
}

// \=================================================================================================  
// FILE: ImpactEventBridgeSystem.cs  
// \=================================================================================================

\[UpdateInGroup(typeof(LateSimulationSystemGroup))\]  
public partial class ImpactEventBridgeSystem : SystemBase  
{  
    private EntityQuery \_impactEventQuery;  
      
    protected override void OnCreate()  
    {  
        \_impactEventQuery \= GetEntityQuery(ComponentType.ReadOnly\<ImpactEvent\>());  
    }  
      
    protected override void OnUpdate()  
    {  
        var impacts \= \_impactEventQuery.ToComponentDataArray\<ImpactEvent\>(Allocator.TempJob);  
        var entities \= \_impactEventQuery.ToEntityArray(Allocator.TempJob);  
          
        for (int i \= 0; i \< impacts.Length; i++)  
        {  
            var impact \= impacts\[i\];  
              
            // Find the ship GameObject  
            if (ProxyManager.Instance.TryGetProxy(impact.Target, out var shipProxy))  
            {  
                // Get damage receiver component  
                var damageReceiver \= shipProxy.GetComponent\<DamageReceiver\>();  
                if (damageReceiver \!= null)  
                {  
                    // Queue async damage calculation  
                    damageReceiver.QueueImpact(new DamageEvent  
                    {  
                        Position \= impact.ImpactPosition,  
                        Normal \= impact.ImpactNormal,  
                        Energy \= impact.ImpactEnergy,  
                        ProjectileVelocity \= impact.ProjectileVelocity  
                    });  
                }  
            }  
              
            // Remove the impact event  
            EntityManager.RemoveComponent\<ImpactEvent\>(entities\[i\]);  
        }  
          
        impacts.Dispose();  
        entities.Dispose();  
    }  
}

// Dummy class to make ImpactEventBridgeSystem compile  
public class DamageReceiver : MonoBehaviour  
{  
    public void QueueImpact(DamageEvent damageEvent)  
    {  
        // This is where you would handle the damage event,  
        // likely queuing it for async processing.  
    }  
}

// Dummy struct to make ImpactEventBridgeSystem compile  
public struct DamageEvent  
{  
    public float3 Position;  
    public float3 Normal;  
    public float Energy;  
    public float3 ProjectileVelocity;  
}

// \=================================================================================================  
// FILE: FireControlSystem.cs  
// \=================================================================================================

public class FireControlSystem : MonoBehaviour  
{  
    \[Header("Performance")\]  
    \[SerializeField\] private int maxCalculationsPerFrame \= 5;  
    \[SerializeField\] private float recalculationInterval \= 0.5f; // seconds  
      
    private Queue\<BallisticCalculationRequest\> \_calculationQueue \= new Queue\<BallisticCalculationRequest\>();  
    private Dictionary\<Turret, float\> \_lastCalculationTime \= new Dictionary\<Turret, float\>();  
      
    void Update()  
    {  
        // Process limited calculations per frame  
        int calculations \= 0;  
        while (\_calculationQueue.Count \> 0 && calculations \< maxCalculationsPerFrame)  
        {  
            var request \= \_calculationQueue.Dequeue();  
            // You would call your ballistic calculation logic here, e.g.:  
            // CalculateBallisticSolution(request);   
            calculations++;  
        }  
          
        // This part is illustrative. You would need to replace \`activeTurrets\`  
        // with your actual collection of turrets that need calculations.  
        /\*  
        foreach (var turret in activeTurrets)  
        {  
            if (Time.time \- \_lastCalculationTime\[turret\] \> recalculationInterval)  
            {  
                \_calculationQueue.Enqueue(new BallisticCalculationRequest(turret));  
                \_lastCalculationTime\[turret\] \= Time.time;  
            }  
        }  
        \*/  
    }  
}

// Dummy structs to make the example compile  
public struct Turret {}  
public struct BallisticCalculationRequest   
{  
    public BallisticCalculationRequest(Turret turret) {}  
}

// \=================================================================================================  
// FILE: EntityProxy.cs  
// \=================================================================================================

public class EntityProxy : MonoBehaviour  
{  
    private Entity \_entity;  
    private World \_world;  
      
    public Entity Entity \=\> \_entity;  
    public bool IsValid \=\> \_world \!= null && \_world.IsCreated && \_world.EntityManager.Exists(\_entity);  
      
    public void Initialize(Entity entity, World world)  
    {  
        \_entity \= entity;  
        \_world \= world;  
        ProxyManager.Instance.RegisterProxy(entity, this);  
    }  
      
    void OnDestroy()  
    {  
        if (ProxyManager.Instance \!= null && IsValid)  
        {  
            ProxyManager.Instance.UnregisterProxy(\_entity);  
        }  
    }  
}

// \=================================================================================================  
// FILE: ProxyManager.cs  
// \=================================================================================================

public class ProxyManager : MonoBehaviour  
{  
    public static ProxyManager Instance { get; private set; }  
      
    private readonly Dictionary\<Entity, EntityProxy\> \_entityToProxy \= new Dictionary\<Entity, EntityProxy\>();  
    private readonly object \_lock \= new object();  
      
    void Awake()  
    {  
        if (Instance \!= null)  
        {  
            Destroy(gameObject);  
            return;  
        }  
        Instance \= this;  
        DontDestroyOnLoad(gameObject);  
    }  
      
    public void RegisterProxy(Entity entity, EntityProxy proxy)  
    {  
        lock (\_lock)  
        {  
            \_entityToProxy\[entity\] \= proxy;  
        }  
    }  
      
    public void UnregisterProxy(Entity entity)  
    {  
        lock (\_lock)  
        {  
            \_entityToProxy.Remove(entity);  
        }  
    }  
      
    public bool TryGetProxy(Entity entity, out EntityProxy proxy)  
    {  
        lock (\_lock)  
        {  
            return \_entityToProxy.TryGetValue(entity, out proxy);  
        }  
    }  
}  
