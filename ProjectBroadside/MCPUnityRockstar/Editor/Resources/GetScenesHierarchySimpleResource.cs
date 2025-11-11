using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Newtonsoft.Json.Linq;

namespace McpUnity.Resources
{
    /// <summary>
    /// Resource for retrieving a simplified hierarchy of all game objects in the Unity scenes.
    /// This provides a lightweight version that includes only essential information (name, instanceId, children).
    /// </summary>
    public class GetScenesHierarchySimpleResource : McpResourceBase
    {
        public GetScenesHierarchySimpleResource()
        {
            Name = "get_scenes_hierarchy_simple";
            Description = "Retrieves a simplified hierarchy of all game objects in the Unity loaded scenes (lightweight version)";
            Uri = "unity://scenes_hierarchy_simple";
        }

        /// <summary>
        /// Fetch a simplified hierarchy of all game objects in the Unity loaded scenes.
        /// </summary>
        /// <param name="parameters">Resource parameters as a JObject (not used)</param>
        /// <returns>A JObject containing the simplified hierarchy of game objects</returns>
        public override JObject Fetch(JObject parameters)
        {
            // Get all game objects in the hierarchy
            JArray hierarchyArray = GetSimpleSceneHierarchy();

            // Create the response
            return new JObject
            {
                ["success"] = true,
                ["message"] = $"Retrieved simplified hierarchy with {hierarchyArray.Count} scene(s)",
                ["hierarchy"] = hierarchyArray
            };
        }

        /// <summary>
        /// Get a simplified hierarchy of all game objects in the Unity loaded scenes.
        /// </summary>
        /// <returns>A JArray containing the simplified hierarchy of game objects</returns>
        private JArray GetSimpleSceneHierarchy()
        {
            JArray scenesArray = new JArray();

            // Get all loaded scenes
            int sceneCount = SceneManager.loadedSceneCount;
            for (int i = 0; i < sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);

                // Create a scene object with minimal info
                JObject sceneObject = new JObject
                {
                    ["name"] = scene.name,
                    ["rootObjects"] = new JArray()
                };

                // Get root game objects in the scene
                GameObject[] rootObjects = scene.GetRootGameObjects();
                JArray rootObjectsInScene = (JArray)sceneObject["rootObjects"];

                foreach (GameObject rootObject in rootObjects)
                {
                    // Add the root object and its children to the array
                    rootObjectsInScene.Add(GameObjectToSimpleJObject(rootObject));
                }

                // Add the scene to the scenes array
                scenesArray.Add(sceneObject);
            }

            return scenesArray;
        }

        /// <summary>
        /// Convert a GameObject to a simplified JObject with its hierarchy.
        /// Only includes essential information: name, instanceId, and children.
        /// </summary>
        /// <param name="gameObject">The GameObject to convert</param>
        /// <returns>A JObject representing the simplified GameObject</returns>
        public static JObject GameObjectToSimpleJObject(GameObject gameObject)
        {
            if (gameObject == null) return null;

            // Add children recursively
            JArray childrenArray = new JArray();
            foreach (Transform child in gameObject.transform)
            {
                childrenArray.Add(GameObjectToSimpleJObject(child.gameObject));
            }

            // Create a JObject for the game object with minimal data
            JObject gameObjectJson = new JObject
            {
                ["name"] = gameObject.name,
                ["instanceId"] = gameObject.GetInstanceID(),
                ["children"] = childrenArray
            };

            return gameObjectJson;
        }
    }
}