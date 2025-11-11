using Unity.Entities;

[System.Serializable]
public struct ShipProperties : IComponentData
{
    public float EnginePower;
    public float RudderEffectiveness;
}