

public class mgb_onclick:mgb
{

    private bool first_click = true; // checks whether the click is the first or the second
    pythonprocess proc = pythonprocess.Pythonprocess();
    
    public void Handle_Button()
    { 
        
       if (first_click)
        {
            base.Generate_click(proc.process);
            //UnityEngine.Debug.Log("yes");
        }

       else
            base.Stop_click(proc.process);
            //UnityEngine.Debug.Log("no");
        first_click = !first_click;
    }



    }

