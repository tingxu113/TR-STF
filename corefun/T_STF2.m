function [L, D, R]=T_STF2(X,r2)
[n1, n2]=size(X);
L=eye(n1,r2);
D=eye(r2,r2);
R=eye(r2,n2);
[L,~]=qr(X*R',0);
[RR,R2]=qr(X'*L,0);
R=RR';
D=L'*X*R'; 
end