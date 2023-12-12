import numpy as np
import math



def hybrid_norm(transition,vec):
        term=vec.clone()
        one=vec.clone()
        one.set(1.)
        term.multiply(term) #r**2
        rt=1/transition/transition #1/transition/transition
        term.scale(rt) #r**2/transition/transition
        term.add(one) # 1+r**2/transition/transition
        term.pow(.5) # (1+r*r/transition/transition)^.5
        c0=term.clone()
        c0.scale_add(one,sc1=1,sc2=-1) #(term-1)
        c0.scacle(transition) # rt*(term-1)
        c1=vec.clone()
        tinv=term.clone()
        tinv.pow(-1)
        c1.multiply(tinv) # c1=rr/term
        c1.scale(1./transition) #c1=rr/rt/term
        c2=one.clone()
        term.pow(-3.)
        c2.multiply(term) #c2=1./term^3
        c2.scale(1./transition) #c2=1./transition/term^3
        return c0,c1,c2
def l1_norm(vec):
    c0=vec.clone()
    c0.multiply(vec)
    c0.pow(.5) # ||r||
    c1=vec.clone()
    mx=max(c1.max(),1e-12)
    c1=c1.scale(1./mx)
    pow=vec.clone()
    pow.multiply(pow)
    pow.scale(-2*1e12) # -r^2 *e^12
    c2=pow.clone()
    c2.set(2.7182818)
    c2.pow(pow)
    c2.scale(2/(1e-6*sqrt(math.pi)))
    return c0,c1,c2
def l2_norm(vec):
    c0=vec.clone()
    c0=c0.multiply(c0)
    c0.scale(.5)
    c1=vec.clone()
    c2=vec.clone()
    c2.set(1.)
    return c0,c1,c2

