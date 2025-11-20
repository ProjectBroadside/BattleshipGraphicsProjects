using Unity.Entities;

public struct ActiveTorpedoTag : IComponentData { }
public struct TorpedoProperties : IComponentData {
    public float MotorForce;
    public float RunTime;
    public float TurnRate;
}