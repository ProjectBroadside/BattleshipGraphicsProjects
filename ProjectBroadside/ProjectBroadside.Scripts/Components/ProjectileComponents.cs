using Unity.Entities;

public struct ActiveProjectileTag : IComponentData { }
public struct PooledProjectileTag : IComponentData { }
public struct BallisticProperties : IComponentData { public float DragFactor; }