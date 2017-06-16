function [ L_cur ] = OML_Core(L_pre, lambda, x_t, x_p, x_q)
% The core algorithm of OPML
a=x_t-x_p;
b=x_t-x_q;
aa= a'*a;
bb=b'*b;
ab=a'*b;
ba=b'*a;


B_tr = lambda*(sum(a.^2) - sum(b.^2));
B_tr2 = lambda^2*(a'*a*sum(a.^2) - a'*b*sum(a.*b) - b'*a*sum(b.*a) + b'*b*sum(b.^2));
eta=1+B_tr;
beta=0.5*(B_tr^2 - B_tr2);


Temp1 = L_pre*a*a' - L_pre*b*b';
Temp2 = aa*L_pre*a*a' - ab*L_pre*a*b' - ba*L_pre*b*a' + bb*L_pre*b*b';
L_cur = L_pre - (eta*lambda)/(eta+beta)*Temp1+ lambda^2/(eta+beta)*Temp2;


end

