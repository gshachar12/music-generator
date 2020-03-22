using System.Collections;
using System.Diagnostics;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public class mgb: MonoBehaviour {
    public Button mgb_button;
    // Update is called once per frame
    public void Generate_click(Process process)
    {
        process.Start();
        UnityEngine.Debug.Log("generating");
    }
    public void Stop_click(Process process)
    {
        process.Kill();
        UnityEngine.Debug.Log("stop");

    }
}
