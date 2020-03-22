using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using UnityEngine.UI;

public class quit: mgb {

    pythonprocess proc = pythonprocess.Pythonprocess();
    public void OnClickButton()


    {
        proc.process.Refresh();  // Important


        if (!proc.process.HasExited)
        {
            Stop_click(proc.process);
        }
        Application.Quit();

    }
}
