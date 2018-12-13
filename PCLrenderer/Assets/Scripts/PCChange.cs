using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class PCChange : MonoBehaviour {

	public string Path = "";
	private static Slider slider;

	private void LoadAllPCL(string path){

	}

	// Use this for initialization
	void Start () {
		slider = gameObject.GetComponent<Slider>();
	}
	
	// Update is called once per frame
	void Update () {

		if(Input.GetButtonDown("Right")){
			slider.value++;
		}

		if(Input.GetButtonDown("Left")){
			slider.value--;
		}

	}
}
