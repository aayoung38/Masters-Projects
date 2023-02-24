package A1;
import java.util.Scanner;

public class Set {

	private int size;
	private String [] data;
	
	public Set(int i)
	{
		data = new String[i];
	}
	
	public boolean contains(String token)
	{
		boolean duplicate = false;
		for (int i =0; i< size; i++)
		{
			if (data[i].equals(token))
			{
				duplicate = true;
				break;
			}
		}
		return duplicate;
	}
	
	/**
	 * Sorts the set.
	 */
	private void sort()
	{
	  for (int i=1;i<size;i++)
	  {
	    String key = data[i];
	    int j = i - 1;
	    while (j>= 0 && key.compareTo(data[j]) <0)
	    {
	      data[j+1] = data[j];
	      j=j-1;
	    }
	    data[j+1] = key;
	  }
	}
	
	/**
	 * Inserts the string into the set
	 * @param s string to insert
	 */
	public void insert(String s)
	{
		if(!contains(s))
		{
			data[size] = s;
			size++;
			sort();
		}
	}
	
	/**
	 * Gets the size of the set
	 * @return size of the set
	 */
	public int getSize()
	{
	  return size;
	}
	
	/**
	 * Gets the specified set element 
	 * @param i index to get set
	 * @return element at index i
	 */
	public String dataAt(int i)
	{
	  return data[i];
	}
	
  /**
   * Outputs the union set between set 1 and set 2
   * @param set2 set to create union set with
   */
	public void union(Set set2)
	{    
	  Set union = new Set(size + set2.getSize());
	  for (int i=0; i<size; i++)
	  {
	    union.insert(data[i]);
	  }
	  for (int i=0; i<set2.getSize();i++)
	  {
	    union.insert(set2.dataAt(i));
	  }
	  System.out.println(union);
	}
	
  /**
   * Outputs the intersection set between set 1 and set 2
   * @param set2 set to create intersection set with
   */
	public void intersection(Set set2)
  {
    Set intersection = new Set(size > set2.getSize() ? size : set2.getSize());
    for (int i=0; i<size; i++)
    {
      if(set2.contains(data[i]))
      {
        intersection.insert(data[i]);
      }
    }
    
    System.out.println(intersection);
  }
	
	/**
	 * Outputs the cartesian set between set 1 and set 2
	 * @param set2 set to create cartesian set with
	 */
	public void cartesian(Set set2)
  {
    Set cartesian = new Set(size * set2.getSize());
    for (int i=0; i<size; i++)
    {
      for(int j=0; j<set2.getSize();j++)
      {
        cartesian.insert(data[i]+" "+set2.dataAt(j));
      }
    }
    
    System.out.println(cartesian);
  }
	
	/**
	 * Creates string representation of set
	 */
	public String toString()
	{
	  StringBuilder out = new StringBuilder();
	  for(int i=0; i<size; i++)
	  {
	    out.append(data[i]);
	    out.append("\n");
	  }
	  return out.toString();
	}
	
	/**
	 * Inputs assignment parameters from standard input.
	 * 
	 * @param sc scanner object
	 * @return set of input parameters
	 */
	public static Set inputSet(Scanner sc)
	{
	  String token=null;
	  Set setTokens=null;
	  
	  if(sc.hasNext())
	  {
	    try 
	    {
	      token = sc.next();
	      int numTokens = Integer.parseInt(token);
	      setTokens = new Set(numTokens);
	      
	      for(int i=0; i< numTokens; i++)
	      {
	        token = sc.next();
	        setTokens.insert(token);
	      }
	      
	    }catch(Exception e)
	    {
	      System.err.println("Could not convert "+token+" to number.");
	    }
	  }
	  return setTokens;
	}
	
	/**
	 * Main
	 * @param args command line args
	 */
	public static void main (String [] args)
	{
		Scanner sc = new Scanner(System.in);
		while(sc.hasNext())
		{
		  System.out.println("---");
		  Set set1 = inputSet(sc);
		  Set set2 = inputSet(sc);
		
		  set1.union(set2);
		  set1.intersection(set2);
		  set1.cartesian(set2);
      System.out.println("---");
		}
	}
}
