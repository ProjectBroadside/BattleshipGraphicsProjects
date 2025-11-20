using System.Collections.Concurrent;
using Unity.Entities;

/// <summary>
/// A thread-safe queue for buffering ship commands from the main thread to the ECS systems.
/// This ensures that commands like steering, throttle, and firing are processed in a structured manner.
/// </summary>
public static class ShipCommandQueue
{
    /// <summary>
    /// Represents a single command to be executed by a ship entity.
    /// </summary>
    public struct ShipCommand
    {
        /// <summary>
        /// The entity representing the ship that the command is for.
        /// </summary>
        public Entity ShipEntity;
        /// <summary>
        /// The type of command to be executed (e.g., Throttle, Rudder).
        /// </summary>
        public CommandType Type;
        /// <summary>
        /// The value associated with the command (e.g., throttle percentage, rudder angle).
        /// </summary>
        public float Value;
    }

    /// <summary>
    /// The internal concurrent queue storing the commands.
    /// </summary>
    private static readonly ConcurrentQueue<ShipCommand> CommandQueue = new ConcurrentQueue<ShipCommand>();

    /// <summary>
    /// Adds a new command to the queue.
    /// </summary>
    /// <param name="command">The ship command to enqueue.</param>
    public static void Enqueue(ShipCommand command)
    {
        CommandQueue.Enqueue(command);
    }

    /// <summary>
    /// Attempts to remove and return the command at the beginning of the queue.
    /// </summary>
    /// <param name="command">When this method returns, contains the dequeued command, if the operation was successful.</param>
    /// <returns>true if a command was successfully removed; otherwise, false.</returns>
    public static bool TryDequeue(out ShipCommand command)
    {
        return CommandQueue.TryDequeue(out command);
    }

    /// <summary>
    /// Gets the current number of commands in the queue.
    /// </summary>
    public static int Count => CommandQueue.Count;
}