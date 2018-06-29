using System.Collections;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public enum PerceptronOutputClass
{
    Red,
    Blue
}

[ExecuteInEditMode]
public class PerceptronLearning : MonoBehaviour {

    public List<TrainingSample> trainingDataset;

    public Vector w;

    [HideInInspector]
    public float W0;

    [HideInInspector]
    public float W1;

    private MeshRenderer quadRenderer;

	// Use this for initialization
	void Start () {
	   quadRenderer = GetComponent<MeshRenderer>();	
	}
	
	// Update is called once per frame
	void Update () {
        
        if (w != null)
        {
            Vector3 localPosition = w.transform.localPosition;

            W0 = localPosition.x;
            W1 = localPosition.y;
        }

        quadRenderer.sharedMaterial.SetFloat("_W0", W0); 
        quadRenderer.sharedMaterial.SetFloat("_W1", W1);
	}
}

#if UNITY_EDITOR
[CustomEditor(typeof(PerceptronLearning))]
public class PerceptronLearningEditor : Editor {

    void OnSceneGUI()
    {
        PerceptronLearning editorObj = target as PerceptronLearning;
        if (editorObj == null) return;

        float newW0 = EditorGUILayout.FloatField("W0", editorObj.W0);
        float newW1 = EditorGUILayout.FloatField("W1", editorObj.W1);

        if (newW0 != editorObj.W0 || newW1 != editorObj.W1)
        {
            editorObj.W0 = newW0;
            editorObj.W1 = newW1;

            Vector3 localPosition = editorObj.w.transform.localPosition;
            editorObj.w.transform.localPosition = new Vector3(newW0, newW1, localPosition.z);
        }      
    }

}
#endif
