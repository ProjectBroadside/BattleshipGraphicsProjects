using Unity.Burst;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Systems;

[UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
[UpdateBefore(typeof(PhysicsSystemGroup))] // Apply our custom forces before the main physics step
[BurstCompile]
public partial struct BallisticsSystem : ISystem
{
    [BurstCompile]
    public void OnCreate(ref SystemState state)
    {
        // Only run the system if there are active projectiles
        state.RequireForUpdate<ActiveProjectileTag>();
        state.RequireForUpdate<PhysicsStep>(); // Changed from PhysicsGravity
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        new BallisticsJob
        {
            DeltaTime = SystemAPI.Time.DeltaTime,
            Gravity = SystemAPI.GetSingleton<PhysicsStep>().Gravity // Changed from PhysicsGravity
        }.ScheduleParallel();
    }
}

[BurstCompile]
public partial struct BallisticsJob : IJobEntity
{
    public float DeltaTime;
    public float3 Gravity;

    // This job runs on each entity that has an ActiveProjectileTag and the other specified components.
    public void Execute(ref PhysicsVelocity velocity, in PhysicsMass mass, in BallisticProperties properties, in ActiveProjectileTag tag)
    {
        // 1. Apply Gravity
        // While the PhysicsSystemGroup also applies gravity, applying it here ensures
        // it's part of the same integration step as our custom forces like drag.
        velocity.Linear += Gravity * DeltaTime;

        // 2. Apply Air Drag
        // This is a simplified model of quadratic drag.
        // Drag Force = -0.5 * C * A * rho * v^2 * v_hat
        // Where C is drag coefficient, A is cross-sectional area, rho is air density.
        // We simplify this into a single 'DragFactor' for gameplay tuning.
        float speed = math.length(velocity.Linear);
        if (speed > 0.01f)
        {
            float3 dragForce = -velocity.Linear * speed * properties.DragFactor;
            
            // Apply as an impulse. For a projectile, we can assume the force is applied at the center of mass.
            velocity.Linear += dragForce / mass.InverseMass * DeltaTime;
        }

        // Future additions could include:
        // - Coriolis Effect: A slight deviation based on projectile velocity and planet's rotation.
        // - Magnus Effect: Lift force due to projectile spin (rifling).
        // - Air Density changes with altitude.
    }
}