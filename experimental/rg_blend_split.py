"""CB5 fixed-row composition with CPU-owned document slots and GPU local slots.

Selection is a frozen CB4 trace, identical for GPU and R placement. This
isolates placement from selection-trajectory changes; it is not an adaptive
online selector. Native CB3 remains untouched and is a separate bridge control.
"""
from contextlib import contextmanager
import torch
from experimental.rg_acceptance import require
from experimental.rg_fp64_accumulation import fp64_parts


def causal(query_positions,key_positions):
    return (key_positions[None,:]<=query_positions[:,None])[None,None]


@contextmanager
def wide_native():
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    original=qwen.eager_attention_forward
    def forward(module,q,k,v,mask,scaling,dropout=0.,**kwargs):
        require(not module.training and not dropout,'inference only')
        output=fp64_parts(q,k[:,:,:0],v[:,:,:0],k,v,None,mask,scaling)[0].to(q.dtype)
        return output.transpose(1,2).contiguous(),None
    qwen.eager_attention_forward=forward
    try:yield
    finally:qwen.eager_attention_forward=original


class SplitBlend:
    def __init__(self,model,input_ids,raw,document_positions,selected_positions,cpu=True,full=False,audit=None):
        require(not model.training and not torch.is_grad_enabled(),'eval inference only')
        require(model.config.model_type=='qwen2' and not getattr(model.config,'use_sliding_window',False),'dense Qwen2 only')
        require(input_ids.ndim==2 and input_ids.shape[0]==1,'batch1 required')
        self.model=model;self.cpu=cpu;self.audit=audit;self.states=[];self.trace=[]
        self.device=input_ids.device;self.length=input_ids.shape[1]
        require(self.length<=model.config.max_position_embeddings,'context overflow')
        self.doc=torch.tensor(document_positions,device=self.device,dtype=torch.long)
        self.selected=torch.tensor(selected_positions,device=self.device,dtype=torch.long)
        for p in (self.doc,self.selected):
            require(torch.equal(p,p.unique(sorted=True)) and bool((p>=0).all()) and bool((p<self.length).all()),'invalid positions')
        require(len(self.selected)>0 and int(self.selected[-1])==self.length-1,'final question row must be fresh')
        if full or not len(self.doc):self.selected=torch.arange(self.length,device=self.device)
        keep=torch.ones(self.length,dtype=torch.bool,device=self.device);keep[self.doc]=False
        require(bool(torch.isin(keep.nonzero().flatten(),self.selected).all()),'uncached rows omitted')
        self.full=full or not len(self.doc)
        if not self.full:
            require(len(raw)==len(model.model.layers),'missing raw layers')
            for k,v in raw:
                require(k.device.type==v.device.type=='cpu','document store must remain CPU-owned')
                require(k.shape==v.shape==(1,model.config.num_key_value_heads,len(self.doc),
                    model.config.hidden_size//model.config.num_attention_heads),'document KV shape mismatch')
                require(k.dtype==v.dtype==model.dtype,'document KV dtype mismatch')
        all_pos=torch.arange(self.length,device=self.device)
        external=all_pos[~torch.isin(all_pos,self.selected)]
        hidden=model.model.embed_tokens(input_ids)
        cos,sin=model.model.rotary_emb(hidden,all_pos[None])
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        for index,layer in enumerate(model.model.layers):
            qp=all_pos if index==0 or self.full else self.selected
            # Layers0/1 project all rows. Later layers project selected rows only.
            projected=all_pos if index<=1 or self.full else self.selected
            residual=hidden;normalized=layer.input_layernorm(hidden);attn=layer.self_attn
            q,k,v=self.project(attn,normalized,cos[:,projected],sin[:,projected])
            if index==1 and not self.full:q=q[:,:,self.selected];residual=residual[:,self.selected]
            if index<=1 or self.full:
                rk,rv=k[:,:,external].contiguous(),v[:,:,external].contiguous()
                if cpu:rk,rv=rk.cpu(),rv.cpu()
                kg,vg=k[:,:,self.selected].contiguous(),v[:,:,self.selected].contiguous()
            else:
                mapping=torch.searchsorted(self.doc,external)
                require(torch.equal(self.doc[mapping],external),'external row is not a document row')
                rk,rv=(value[:,:,mapping.cpu()].contiguous() for value in raw[index])
                # Native RoPE operations on CPU; GPU/native remap checked separately by auditor.
                rc,rs=cos[:,external].cpu(),sin[:,external].cpu()
                rk=apply_rotary_pos_emb(rk,rk,rc,rs)[1]
                if not cpu:rk,rv=rk.to(self.device),rv.to(self.device)
                kg,vg=k,v
            state=dict(rk=rk,rv=rv,kg=kg,vg=vg,rp=external.clone(),gp=self.selected.clone())
            self.states.append(state)
            output=self.attend(q,qp,state,attn.scaling,index,'prefill',0)
            hidden=residual+attn.o_proj(output.reshape(1,len(qp),-1))
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
            self.trace.append(dict(layer=index,projected_rows=len(projected),query_positions=qp.tolist(),
                                   local_positions=self.selected.tolist(),external_positions=external.tolist(),
                                   external_device=rk.device.type,mask='absolute-causal'))
        self.logits=model.lm_head(model.model.norm(hidden))[:,-1]
        self.check_ownership()

    @staticmethod
    def project(attn,hidden,cos,sin):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        shape=(*hidden.shape[:-1],-1,attn.head_dim)
        q=attn.q_proj(hidden).view(shape).transpose(1,2)
        k=attn.k_proj(hidden).view(shape).transpose(1,2)
        v=attn.v_proj(hidden).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        return q,k,v

    def attend(self,q,qp,state,scale,layer,phase,step):
        mr,mg=causal(qp,state['rp']),causal(qp,state['gp'])
        output=fp64_parts(q,state['rk'],state['rv'],state['kg'],state['vg'],mr,mg,scale)[0]
        if self.audit:self.audit(q,state,mr,mg,scale,output,layer,phase,step)
        return output.to(q.dtype).transpose(1,2).contiguous()

    def check_ownership(self):
        for s in self.states:
            require(s['rk'].device.type==s['rv'].device.type==('cpu' if self.cpu else self.device.type),'external ownership changed')
            require(s['kg'].device==s['vg'].device==self.device,'local ownership changed')
            require(not bool(torch.isin(s['gp'],s['rp']).any()),'persistent external GPU duplicate')
            require(torch.equal(torch.cat((s['gp'],s['rp'])).sort().values,
                                torch.arange(self.length,device=self.device)),'missing/duplicate key positions')

    def decode(self,token,step):
        require(self.length<self.model.config.max_position_embeddings,'decode context overflow')
        qp=torch.tensor([self.length],device=self.device)
        hidden=self.model.model.embed_tokens(torch.tensor([[token]],device=self.device))
        cos,sin=self.model.model.rotary_emb(hidden,qp[None])
        for index,(layer,state) in enumerate(zip(self.model.model.layers,self.states,strict=True)):
            residual=hidden;attn=layer.self_attn
            q,k,v=self.project(attn,layer.input_layernorm(hidden),cos,sin)
            state['kg']=torch.cat((state['kg'],k),dim=2);state['vg']=torch.cat((state['vg'],v),dim=2)
            state['gp']=torch.cat((state['gp'],qp))
            output=self.attend(q,qp,state,attn.scaling,index,'decode',step)
            hidden=residual+attn.o_proj(output.reshape(1,1,-1))
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
        self.length+=1;self.check_ownership()
        self.logits=self.model.lm_head(self.model.model.norm(hidden))[:,-1]
        return self.logits
