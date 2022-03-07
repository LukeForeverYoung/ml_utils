from sys import prefix


class TreeNode():
    def __init__(self,s,v=None) -> None:
        self.s=s
        self.v=v
        self.child={}



def dfs(t,depth=-1):
    if depth!=-1:
        print(f"{'-'*(depth)}{t.s}")
    for child_name,child_node in t.child.items():
        dfs(child_node,depth+1)

class Tree(TreeNode):
    def __init__(self) -> None:
        super().__init__('')
    
    def insert(self,s,v=None):
        s_list=s.split('.')
        node=self
        for si,item in enumerate(s_list):
            if item not in node.child:
                node.child[item]=TreeNode(item)
            node=node.child[item]
        node.v=v
    


            
        
