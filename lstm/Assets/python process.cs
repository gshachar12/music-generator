using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
public class pythonprocess {

    private static pythonprocess script = null; // singleton object 
    
    private string generator = @"c:\project\servers.py {0} {1}"; // the python music generator
    private string python = @"c:\rnn\Scripts\python.exe"; // python file
    public Process process; // process for running the generator file
    public ProcessStartInfo startInfo = new ProcessStartInfo(); // process information
 
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
    public pythonprocess()
    {
        process = new Process();
        Process_Initialization(process); // initialize the script parameters

    }
    public Process new_process(string bpm = "100", string scale = "c")
    { Process proc = new Process();
        Process_Initialization(proc, bpm, scale);
      
        return proc;
    }
    public void Process_Initialization(Process p, string bpm = "100", string scale = "c")
    {
        // initializing parameters for python process initialization
        startInfo.WindowStyle = ProcessWindowStyle.Hidden;
        startInfo.FileName = python;
        startInfo.Arguments = string.Format("{0} {1} {2}", generator, bpm,scale);
        
        UnityEngine.Debug.Log(startInfo.Arguments);
        //startInfo.UseShellExecute = false;
        //startInfo.RedirectStandardOutput = true;

        p.StartInfo = startInfo;
        p.Start();
        

    }
    // static method to create instance of Singleton class 
    public static pythonprocess Pythonprocess()
    {
        if (script== null)
            script = new pythonprocess();

        return script;
    }



}
