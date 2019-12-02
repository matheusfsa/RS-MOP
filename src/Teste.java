import java.util.ArrayList;

import org.json.simple.JSONObject;

import requests.Request;

public class Teste {
	public static void main(String[] args) {
		JSONObject json = new JSONObject();
		json.put("user", 1);
		JSONObject res = Request.execute("http://127.0.0.1:5000/min", json);
		ArrayList<Double> min = (ArrayList<Double>)res.get("response");
		System.out.println(min.get(2));
	}	
}
