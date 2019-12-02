package requests;

import java.util.ArrayList;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import com.fasterxml.jackson.databind.ObjectMapper;

public class Request {
	
	public static JSONObject execute(String url, JSONObject data) {
		HttpClient httpClient = HttpClientBuilder.create().build(); //Use this instead 
		try {

		    HttpPost request = new HttpPost(url);
		    StringEntity params =new StringEntity(data.toString());
		    request.addHeader("content-type", "application/json");
		    request.setEntity(params);
		    HttpResponse response = httpClient.execute(request);
		    String stringToParse = EntityUtils.toString(response.getEntity());
		    JSONParser parser = new JSONParser();
		    JSONObject json = (JSONObject) parser.parse(stringToParse);
		    return json;
		    

		}catch (Exception ex) {
			ex.printStackTrace();
		}
		return null;
	}
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		ArrayList<Double> solucao = new ArrayList<Double>();
		solucao.add(2.0);
		solucao.add(3.4);
		JSONObject json = new JSONObject();
		
		ArrayList<ArrayList<Double>> solucoes = new ArrayList<ArrayList<Double>>();
		
		solucoes.add(solucao);
		json.put("solucoes", solucoes);
		System.out.println(json);
		JSONObject res = Request.execute("http://127.0.0.1:5000/", json);
		System.out.println(res.get("solucoes"));
		System.out.println((ArrayList<ArrayList<Double>>)res.get("solucoes"));
	}
}
