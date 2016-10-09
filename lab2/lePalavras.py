import string
file=open("file.txt", "r")

dicionario={}

 
for numLine,line in enumerate(file,1):
    contaPalavra = 1
    
    for num,word in enumerate(line.split(),0):
        
        # Ola e ola so contam como 1 palavra
        word = word.lower()
        
        # retira pontuaçao, para retirar numeros fazer + string.numbers
        word = word.strip(string.punctuation)

        
        # ja existe no dicionario   
        if word in dicionario:
                #if line.split()[num]==line.split()[num-1]:
                    #contaPalavra+=1
                contaPalavra+=1
                dicionario[word].append([numLine,contaPalavra])
                
        # ainda nao existe no dicionario       
        else:
                
                dicionario[word] = [[numLine,1]] 

        
        if num == 0:
            
            contaPalavra = 1
        elif line.split()[num] != line.split()[num-1]:
            
            
            contaPalavra = 1
            
        
 #        conta apenas o nr de vezes que cada palavra aparece
#        try:
#                dicionario[word] += 1
#
##       if it's not this is the first time we are adding it
#        except:
#                dicionario[word] = 1
                
print dicionario

file.close()

   




