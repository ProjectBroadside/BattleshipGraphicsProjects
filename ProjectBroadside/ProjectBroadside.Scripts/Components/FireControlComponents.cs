using Unity.Entities;

public struct FireControlComponent : IComponentData
{
    public Entity target;
    public bool isFiring;
}