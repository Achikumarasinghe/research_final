
#!/usr/bin/env Rscript
temp = read.csv("dataframe.csv",sep=",")
print(temp)

library(nFCA)  
data("nfca_example", package = "nFCA") 
#nfca_example
#nfca(data = nfca_example)
nfca(data = temp,type=1)

