import{p as B}from"./C78Q6e4S.js";import{_ as b,L as m,O as C,j as S,l as w,d as D,c as T,u as P,v as z,g as E,b as F,J as A,M as W,x as _}from"./Dz7d4_v9.js";import{p as L}from"./M_RCUzsg.js";var M=W.packet,u,v=(u=class{constructor(){this.packet=[],this.setAccTitle=D,this.getAccTitle=T,this.setDiagramTitle=P,this.getDiagramTitle=z,this.getAccDescription=E,this.setAccDescription=F}getConfig(){const t=m({...M,...A().packet});return t.showBits&&(t.paddingY+=10),t}getPacket(){return this.packet}pushWord(t){t.length>0&&this.packet.push(t)}clear(){_(),this.packet=[]}},b(u,"PacketDB"),u),N=1e4,O=b((e,t)=>{B(e,t);let r=-1,s=[],n=1;const{bitsPerRow:l}=t.getConfig();for(let{start:a,end:i,bits:d,label:c}of e.blocks){if(a!==void 0&&i!==void 0&&i<a)throw new Error(`Packet block ${a} - ${i} is invalid. End must be greater than start.`);if(a??=r+1,a!==r+1)throw new Error(`Packet block ${a} - ${i??a} is not contiguous. It should start from ${r+1}.`);if(d===0)throw new Error(`Packet block ${a} is invalid. Cannot have a zero bit field.`);for(i??=a+(d??1)-1,d??=i-a+1,r=i,w.debug(`Packet block ${a} - ${r} with label ${c}`);s.length<=l+1&&t.getPacket().length<N;){const[p,o]=Y({start:a,end:i,bits:d,label:c},n,l);if(s.push(p),p.end+1===n*l&&(t.pushWord(s),s=[],n++),!o)break;({start:a,end:i,bits:d,label:c}=o)}}t.pushWord(s)},"populate"),Y=b((e,t,r)=>{if(e.start===void 0)throw new Error("start should have been set during first phase");if(e.end===void 0)throw new Error("end should have been set during first phase");if(e.start>e.end)throw new Error(`Block start ${e.start} is greater than block end ${e.end}.`);if(e.end+1<=t*r)return[e,void 0];const s=t*r-1,n=t*r;return[{start:e.start,end:s,label:e.label,bits:s-e.start},{start:n,end:e.end,label:e.label,bits:e.end-n}]},"getNextFittingBlock"),x={parser:{yy:void 0},parse:b(async e=>{const t=await L("packet",e),r=x.parser?.yy;if(!(r instanceof v))throw new Error("parser.parser?.yy was not a PacketDB. This is due to a bug within Mermaid, please report this issue at https://github.com/mermaid-js/mermaid/issues.");w.debug(t),O(t,r)},"parse")},j=b((e,t,r,s)=>{const n=s.db,l=n.getConfig(),{rowHeight:a,paddingY:i,bitWidth:d,bitsPerRow:c}=l,p=n.getPacket(),o=n.getDiagramTitle(),h=a+i,g=h*(p.length+1)-(o?0:a),k=d*c+2,f=C(t);f.attr("viewBox",`0 0 ${k} ${g}`),S(f,g,k,l.useMaxWidth);for(const[y,$]of p.entries())I(f,$,y,l);f.append("text").text(o).attr("x",k/2).attr("y",g-h/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),I=b((e,t,r,{rowHeight:s,paddingX:n,paddingY:l,bitWidth:a,bitsPerRow:i,showBits:d})=>{const c=e.append("g"),p=r*(s+l)+l;for(const o of t){const h=o.start%i*a+1,g=(o.end-o.start+1)*a-n;if(c.append("rect").attr("x",h).attr("y",p).attr("width",g).attr("height",s).attr("class","packetBlock"),c.append("text").attr("x",h+g/2).attr("y",p+s/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(o.label),!d)continue;const k=o.end===o.start,f=p-2;c.append("text").attr("x",h+(k?g/2:0)).attr("y",f).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",k?"middle":"start").text(o.start),k||c.append("text").attr("x",h+g).attr("y",f).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(o.end)}},"drawWord"),G={draw:j},H={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},J=b(({packet:e}={})=>{const t=m(H,e);return`
	.packetByte {
		font-size: ${t.byteFontSize};
	}
	.packetByte.start {
		fill: ${t.startByteColor};
	}
	.packetByte.end {
		fill: ${t.endByteColor};
	}
	.packetLabel {
		fill: ${t.labelColor};
		font-size: ${t.labelFontSize};
	}
	.packetTitle {
		fill: ${t.titleColor};
		font-size: ${t.titleFontSize};
	}
	.packetBlock {
		stroke: ${t.blockStrokeColor};
		stroke-width: ${t.blockStrokeWidth};
		fill: ${t.blockFillColor};
	}
	`},"styles"),X={parser:x,get db(){return new v},renderer:G,styles:J};export{X as diagram};
