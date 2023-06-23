function str=disp(tr,name)
%---------------------------

if ~exist('name','var')
    name = 'ans';
end
r=tr.r; n=tr.n; 
d=tr.d;
str=[];
str=[str,sprintf('%s is a %d-dimensional TR-tensor, ranks and mode sizes: \n',name,d)];
%fprintf('r(1)=%d \n', r(1));
for i=1:d
    str=[str,sprintf('r(%d)=%d \t n(%d)=%d \n',i,r(i),i,n(i))];
%   fprintf(' \t n(%d)=%d \n',i,n(i));
%   fprintf('r(%d)=%d \n',i+1,r(i+1));
end

 if ( nargout == 0 )
    fprintf(str);
 end
end

