using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class parameters : MonoBehaviour {

    pythonprocess proc = new pythonprocess();
    public void OnbpmChange(float bpm)
    {// change the bpm value, and pass the new bpm to the generator
        //Debug.Log("bpm" + bpm);
        proc.process=proc.new_process( bpm.ToString()); // set new params
        Debug.Log("done");
    }
}
