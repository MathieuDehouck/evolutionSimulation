"""
..
  Representing a tree as SVG graph.
  Mathieu Dehouck
  06/2024
"""



def layout(node):
    """
    computes the position of the descendencts of a node recursively to make a tree layout
    """

    if node.is_leaf():
        return {node.index:(0, node.depth)}, [(0, node.depth)], [(0, node.depth)] # dictionary of positions, left most, right most

    else:
        lays = [layout(ch) for ch in node.children]
        if len(lays) == 1:
            lay, left, right = lays[0]
            #print(lay, node.children)
            pos = lay[node.children[0].index][0], node.depth
            lay[node.index] = pos
            
            return lay, [pos] + left, [pos] + right
        
        else:
            (llay, lleft, lright) = lays[0]
            lays = lays[1:]

            while lays != []:
                (rlay, rleft, rright) = lays[0]
                lays = lays[1:]

                overlap = 0
                for lr, rl in zip(lright, rleft):
                    overlap = max(overlap, lr[0] - rl[0])
                    #print(lr, rl, overlap)

                delta = overlap + 1

                for k, (x, y) in rlay.items():
                    llay[k] = x+delta, y

                rright = [(x+delta, y) for (x,y) in rright]
                if len(lleft) > len(rright):
                    rright += lleft[len(rright):]
                else:
                    lleft += rright[len(lleft):]

                lright = rright

            pos = (lleft[0][0] + rright[0][0]) / 2, node.depth
            llay[node.index] = pos

            return llay, [pos] + lleft, [pos] + rright



K = 75
def toSVG(tree, fname):

    fout = open(fname+'.svg', 'w')
    
    lay, _, _ = layout(tree.root)
    X = max([x for x, _ in lay.values()])
    Y = max([y for _, y in lay.values()])

    print('<svg version="1.1" width="'+str(K*(X+2))+'" height="'+str(50*(Y+2))+'" xmlns="http://www.w3.org/2000/svg">', file=fout)
    print('<rect width="100%" height="100%" fill="white"/>', file=fout)


    w = 40
    for node in tree.depth_first():
        x, y = lay[node.index]
        for ch in node.children:
            xx, yy = lay[ch.index]
            print('<line x1="'+str((x+1)*K)+'" x2="'+str((xx+1)*K)+'" y1="'+str((y+1)*50)+'" y2="'+str((yy+1)*50)+'" stroke="black"/>', file=fout)

        print('<rect x="'+str((x+1)*K-w/2)+'" y="'+str((y+1)*50-10)+'" width="'+str(w)+'" height="20" fill="white" stroke="black"/>', file=fout)
        #print('<text x="'+str((x+1)*K-5)+'" y="'+str((y+1)*50+4)+'">'+str(node.index)+'</text>', file=fout)
        print('<text x="'+str((x+1)*K-10)+'" y="'+str((y+1)*50+4)+'" fill="red">'+str(node.change)+'</text>', file=fout)
            

    print('</svg>', file=fout)
