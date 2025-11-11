using Unity.Entities;
using Unity.Physics;
using UnityEngine;

public class ProjectileAuthoring : MonoBehaviour {
    [Header("Ballistic Properties")]
    public float massInKg = 50.0f;
    public float dragFactor = 0.002f;
    // The PhysicsShapeAuthoring component should be added to the same GameObject in the editor.

    class Baker : Baker<ProjectileAuthoring> {
        public override void Bake(ProjectileAuthoring authoring) {
            var entity = GetEntity(TransformUsageFlags.Dynamic);
            
            AddComponent(entity, new BallisticProperties { DragFactor = authoring.dragFactor });
            
            // The PhysicsMass component is configured from the PhysicsShapeAuthoring component
            // which should be on the same GameObject. We just need to set the mass value.
            // The old PhysicsMassOverride is replaced by setting the mass directly in the PhysicsMass component.
            AddComponent(entity, new PhysicsMass { 
                InverseMass = 1.0f / authoring.massInKg,
                InverseInertia = new Unity.Mathematics.float3(1,1,1) // Placeholder, should be calculated from collider
            });
        }
    }
}