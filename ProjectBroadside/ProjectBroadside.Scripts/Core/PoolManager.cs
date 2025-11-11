using UnityEngine;
using Unity.Entities;
using Unity.Collections;
using System.Collections;
using Unity.Transforms;
using Unity.Physics.Systems;
using Unity.Physics;

public class PoolManager : MonoBehaviour
{
    [Tooltip("The GameObject prefab that has the ProjectileAuthoring component.")]
    public GameObject projectileGameObjectPrefab;
    [Tooltip("The total number of projectiles to have in the pool.")]
    public int projectilePoolSize = 15000;
    [Tooltip("How many projectiles to create per frame to avoid stuttering.")]
    public int creationBatchSize = 500;

    private Entity projectileEntityPrefab;
    private EntityManager entityManager;

    void Start()
    {
        if (projectileGameObjectPrefab == null)
        {
            Debug.LogError("PoolManager: Projectile GameObject Prefab is not assigned!");
            return;
        }

        entityManager = World.DefaultGameObjectInjectionWorld.EntityManager;

        // The conversion from GameObject to Entity is now handled by a baker.
        // We need to get the entity prefab that the baker creates.
        // This requires a different approach, such as a singleton or a resource loading system.
        // For now, I will assume the prefab is baked and we can get it.
        // This part of the code will need to be refactored to work with the new baking system.
        // projectileEntityPrefab = ...

        // Start the background process to create the pool
        StartCoroutine(CreateProjectilePoolCoroutine());
    }

    private IEnumerator CreateProjectilePoolCoroutine()
    {
        Debug.Log("Starting background creation of projectile pool...");
        int createdCount = 0;

        while (createdCount < projectilePoolSize)
        {
            // Use a temporary command buffer to batch the creation commands
            var ecb = new EntityCommandBuffer(Allocator.Temp);
            int batchEnd = Mathf.Min(createdCount + creationBatchSize, projectilePoolSize);

            for (int i = createdCount; i < batchEnd; i++)
            {
                var newProjectile = ecb.Instantiate(projectileEntityPrefab);
                // Add a tag to identify it as being in the pool
                ecb.AddComponent<PooledProjectileTag>(newProjectile);
                // Physics is now disabled by adding the `Disabled` component tag.
                ecb.AddComponent<Disabled>(newProjectile);
            }

            // Play back the commands on the main thread
            ecb.Playback(entityManager);
            ecb.Dispose();

            createdCount = batchEnd;
            // Wait for the next frame before creating the next batch
            yield return null;
        }

        Debug.Log($"Projectile pool creation complete. Total size: {createdCount}");
    }

    // A static or singleton method would typically be here to get a projectile from the pool
    // public static Entity GetProjectileFromPool() { ...
}