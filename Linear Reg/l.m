function ret=h(x,theta)
    ret=theta(1);
    for i=1:length(x)
        ret=ret+x(i)*theta(i+1);
    end
end
function ret=J(x,theta)
    ret=theta(1);
    for i=1:length(x)
        ret=ret+x(i)*theta(i+1);
    end
end