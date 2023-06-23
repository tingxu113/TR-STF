function [t,node] = tensor_ring(varargin)


% Tensor Ring toolbox

if (nargin == 0)
    t.d    = 0;
    t.r    = 0;
    t.n    = 0;
    t.node = 0;                    % empty tensor
    t = class(t, 'tensor_ring');
    return;
end

ip = inputParser;
ip.addParamValue('Tol', 1e-6, @isscalar);
ip.addParamValue('Alg', 'SVD', @ischar);
ip.addParamValue('Rank', [], @ismatrix);
ip.addParamValue('MaxIter', 20, @isscalar);
ip.parse(varargin{2:end});

Tol = ip.Results.Tol;
Alg = ip.Results.Alg;
Rank = ip.Results.Rank;
MaxIter = ip.Results.MaxIter;


rng('default');

if is_array(varargin{1})
    t=tensor_ring;
  if strcmp(Alg,'STF')
        c=varargin{1};
        n = size(c);
        n = n(:);
        d = numel(n);
        node=cell(1,d);
        r=Rank;
        for i=1:d-1
            if i==1            
                c=reshape(c,[n(1),prod(n(2:d))]);
                [L, D, R]=T_STF1(c,r(1),r(2));                       
                node{1}=permute(reshape(L,[n(1),r(2),r(1)]),[3,1,2]);
                c=permute(reshape(D*R,[r(2),r(1),prod(n(2:d))]),[1, 3, 2]);
          else             
                c=reshape(c,[r(i)*n(i),prod(n((i+1):d))*r(1)]);
                [L, D, R]=T_STF2(c,r(i+1));     
                node{i}=reshape(L,[r(i),n(i),r(i+1)]);
                c=reshape(D*R,[r(i+1),prod(n((i+1):d)),r(1)]);
           end
       end
        node{d} = c;
        t.node=node;
        t.d=d;
        t.n=n;
        t.r=r;
        return;
  end
end

if is_array(varargin{1})
    t=tensor_ring;
    if strcmp(Alg,'SVD')
        c=varargin{1};
        n = size(c);
        n = n(:);
        d = numel(n);
        node=cell(1,d);
        r = ones(d,1);
        ep=Tol/sqrt(d);
        for i=1:d-1
            if i==1
                c=reshape(c,[n(i),numel(c)/n(i)]);
                [u,s,v]=svd(c,'econ');
                s=diag(s);
                rc=my_chop2(s,sqrt(2)*ep*norm(s));          
                temp=cumprod(factor(rc));
                [~,idx]=min(abs(temp-sqrt(rc)));         
                r(i+1)=temp(idx); r(i)=rc/r(i+1);
                u=u(:,1:r(i)*r(i+1));
                u=reshape(u,[n(i),r(i+1),r(i)]);
                node{i}=permute(u,[3,1,2]);
                s=s(1:r(i)*r(i+1));
                v=v(:,1:r(i)*r(i+1));
                v=v*diag(s);
                v=v';
                v=reshape(v,[r(i+1),r(i),prod(n(2:end))]);
                c=permute(v,[1,3,2]);
            else
                m=r(i)*n(i); c=reshape(c,[m,numel(c)/m]);
                [u,s,v]=svd(c,'econ');
                s=diag(s); r1=my_chop2(s,ep*norm(s));
                r(i+1)=max(r1,1);
                u=u(:,1:r(i+1));
                node{i}=reshape(u,[r(i),n(i),r(i+1)]);
                v=v(:,1:r(i+1)); s=s(1:r(i+1));
                v=v*diag(s);
                c=v';
            end
        end
        node{d} = reshape(c,[r(d),n(d),r(1)]);
        t.node=node;
        t.d=d;
        t.n=n;
        t.r=r;
        return;
    end
      
    
    
    if strcmp(Alg,'ALS')
        maxit=MaxIter;
        c=varargin{1};
        n = size(c);
        n = n(:);
        d = numel(n);
        node=cell(1,d);
        r=Rank(:);
        for i=1:d-1
            node{i}=randn(r(i),n(i),r(i+1));
        end
        node{d}=randn(r(d),n(d),r(1));
        od=[1:d]';
        err=1;
        for it=1:maxit
            err0=err;
            if it>1
                c=shiftdim(c,1);
                od=circshift(od,-1);
            end
            c=reshape(c,n(od(1)),numel(c)/n(od(1)));
            b=node{od(2)};
            for k=3:d
                j=od(k);
                br=node{j};
                br=reshape(br,[r(j),numel(br)/r(j)]);
                b=reshape(b,[numel(b)/r(j),r(j)]);
                b=b*br;
            end
            b=reshape(b,[r(od(2)),prod(n(od(2:end))),r(od(1))]);
            b=permute(b,[1,3,2]);
            b=reshape(b,[r(od(2))*r(od(1)), prod(n(od(2:end)))]);
            a=c/b;
            err=norm(c-a*b,'fro')/norm(c(:));
            a=reshape(a,[n(od(1)),r(od(2)),r(od(1))]);
            node{od(1)}=permute(a,[3,1,2]);
            s=norm(node{od(1)}(:));
            node{od(1)}=node{od(1)}./s;
            
         %  fprintf('it:%d, err=%f\n',it,err);
            if abs(err0-err)<=1e-3 && it>=2*d && err<=Tol
                break;
            end
            c=reshape(c,n(od)');
        end
        node{od(1)}=node{od(1)}.*s;
        t.node=node;
        t.d=d;
        t.n=n;
        t.r=r;
        return;
    end
    
    if strcmp(Alg,'ALSAR')
        warning('off');
        swithch=0;
        c=varargin{1};
        n = size(c);
        n = n(:);
        d = numel(n);
        % Adjustable parameters
        maxit=MaxIter;   %10
        ratio=0.01/d;  %0.01/d
        
        node=cell(1,d);
        r=ones(d,1);
        for i=1:d-1
            node{i}=randn(r(i),n(i),r(i+1));
        end
        node{d}=randn(r(d),n(d),r(1));
        od=[1:d]';
        for it=1:maxit
            if it>1
                c=shiftdim(c,1);
                od=circshift(od,-1);
            end
            
            c=reshape(c,n(od(1)),numel(c)/n(od(1)));
            b=node{od(2)};
            for k=3:d
                j=od(k);
                br=node{j};
                br=reshape(br,[r(j),numel(br)/r(j)]);
                b=reshape(b,[numel(b)/r(j),r(j)]);
                b=b*br;
            end
            b=reshape(b,[r(od(2)),prod(n(od(2:end))),r(od(1))]);
            b=permute(b,[1,3,2]);
            b=reshape(b,[r(od(2))*r(od(1)), prod(n(od(2:end)))]);
            a=c/b;
            err0=norm(c-a*b,'fro')/norm(c(:));
            a=reshape(a,[n(od(1)),r(od(2)),r(od(1))]);
            node{od(1)}=permute(a,[3,1,2]);
       
            r(od(2))=r(od(2))+1;
            node{od(2)}(r(od(2)),:,:)=mean(node{od(2)}(:))+std(node{od(2)}(:)).*randn(n(od(2)),r(od(3)));
            b=node{od(2)};
            for k=3:d
                j=od(k);
                br=node{j};
                br=reshape(br,[r(j),numel(br)/r(j)]);
                b=reshape(b,[numel(b)/r(j),r(j)]);
                b=b*br;
            end
            b=reshape(b,[r(od(2)),prod(n(od(2:end))),r(od(1))]);
            b=permute(b,[1,3,2]);
            b=reshape(b,[r(od(2))*r(od(1)), prod(n(od(2:end)))]);
            a=c/b;
            err1=norm(c-a*b,'fro')/norm(c(:));
            if (err0-err1)/(err0) > ratio*(err0-Tol)/err0 && err0>Tol
                a=reshape(a,[n(od(1)),r(od(2)),r(od(1))]);
                node{od(1)}=permute(a,[3,1,2]);
                err0 =err1;
                swithch =0;
            else
                node{od(2)}(r(od(2)),:,:)=[];
                r(od(2))=r(od(2))-1;
                swithch=1;
            end
            
            s=norm(node{od(1)}(:));
            node{od(1)}=node{od(1)}./s;
           % fprintf('it:%d, err=%f\n',it,err0);
            if err0<=Tol && it>=2*d && swithch ==1
                break;
            end
            c=reshape(c,n(od)');
        end
        node{od(1)}=node{od(1)}.*s;
        t.node=node;
        t.d=d;
        t.n=n;
        t.r=r;
        return;
    end
    
    if strcmp(Alg,'BALS')
        maxit=MaxIter;
        c=varargin{1};
        n = size(c);
        n = n(:);
        d = numel(n);
        node=cell(1,d);
        r=ones(d,1);
        for i=1:d-1
            node{i}=randn(r(i),n(i),r(i+1));
        end
        node{d}=randn(r(d),n(d),r(1));
        od=[1:d]';
        for it=1:maxit
            if it>1
                c=shiftdim(c,1);
                od=circshift(od,-1);
            end    
            c=reshape(c,n(od(1))*n(od(2)),numel(c)/(n(od(1))*n(od(2))));
            b=node{od(3)};
            for k=4:d
                j=od(k);
                br=node{j};
                br=reshape(br,[r(j),numel(br)/r(j)]);
                b=reshape(b,[numel(b)/r(j),r(j)]);
                b=b*br;
            end
            b=reshape(b,[r(od(3)),prod(n(od(3:end))),r(od(1))]);
            b=permute(b,[1,3,2]);
            b=reshape(b,[r(od(3))*r(od(1)), prod(n(od(3:end)))]);
            a=c/b;
            err0=norm(c-a*b,'fro')/norm(c(:));   
            a=reshape(a,[n(od(1)),n(od(2)),r(od(3)),r(od(1))]);
            a=permute(a,[4 1 2 3]);
            a=reshape(a, [r(od(1))*n(od(1)), n(od(2))*r(od(3))]);
            [u,s,v]=svd(a,'econ');
            s=diag(s);       
            r1=my_chop2(s,max([1/sqrt(d)*err0, Tol/sqrt(d)])*norm(s));     
            r(od(2))= max(r1,1);     
            u=u(:,1:r(od(2)));
            s=s(1:r(od(2)));
            v=v(:,1:r(od(2)));
            v=v*diag(s); v=v';
            node{od(1)}=reshape(u,[r(od(1)),n(od(1)),r(od(2))]);
            node{od(2)}=reshape(v,[r(od(2)),n(od(2)),r(od(3))]);
          %  fprintf('it:%d, err=%6f\n',it,err0);
            if err0<=Tol && it>=2*d
                anew=reshape(node{od(1)},[r(od(1))*n(od(1)),r(od(2))])*reshape(node{od(2)},[r(od(2)),n(od(2))*r(od(3))]);
                anew=reshape(anew,[r(od(1)),n(od(1)), n(od(2)),r(od(3))]);
                anew=permute(anew,[2 3 4 1]);
                anew=reshape(anew,[n(od(1))*n(od(2)),r(od(3))*r(od(1))]);
                err1=norm(c-anew*b,'fro')/norm(c(:));
                if err1<=Tol
                    break;
                end
            end
            c=reshape(c,n(od)');
        end
        t.node=node;
        t.d=d;
        t.n=n;
        t.r=r;
        return;
    end  
  
end 