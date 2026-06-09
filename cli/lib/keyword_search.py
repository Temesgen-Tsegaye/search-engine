
from cli.lib.search_util import load_movies

def seach_command(query,n_result):
    data=load_movies()
    res=[]
    for movie in data:
        if(query in movie["title"]):
            res.append(movie)
            if len(res)>n_result:
                break
    
    return res

        
