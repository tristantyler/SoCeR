

def read_my_name_for_test(file_name,skippedline=2):
    z_coord=[]
    ## this is a comment
    with open(file_name) as f:
        for line in f.readlines():
            if(skippedline==0):
                a=line.split()
                z_coord.append(a[3])
            else:
                skippedline-=1
    return z_coord
##if __name__== "__main__":
##  main()


