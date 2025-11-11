using Unity.Entities;

/// <summary>
/// Bridge system for handling ship command communication between MonoBehaviour and DOTS systems.
/// This system processes ship commands from the ShipCommandQueue and translates them into ECS actions.
/// </summary>
public partial class ShipCommandBridgeSystem : SystemBase
{
    protected override void OnUpdate()
    {
        // Process ship commands from the queue
        while (ShipCommandQueue.TryDequeue(out var shipCommand))
        {
            // Process the ship command based on its type
            switch (shipCommand.Type)
            {
                case CommandType.Move:
                    // TODO: Handle movement command
                    break;
                case CommandType.Attack:
                    // TODO: Handle attack command
                    break;
                case CommandType.Follow:
                    // TODO: Handle follow command
                    break;
            }
        }
    }
}