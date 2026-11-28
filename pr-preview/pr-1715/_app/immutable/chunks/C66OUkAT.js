import{p as at}from"./C78Q6e4S.js";import{a0 as T,S as G,be as rt,g as nt,b as it,c as ot,d as st,v as lt,u as ct,_ as g,l as B,e as ut,L as dt,O as gt,aa as pt,j as ht,x as ft,M as mt}from"./Dz7d4_v9.js";import{p as vt}from"./M_RCUzsg.js";import{d as q}from"./CeiKDhVs.js";import{o as xt}from"./CmKTTxBW.js";function St(t,n){return n<t?-1:n>t?1:n>=t?0:NaN}function yt(t){return t}function wt(){var t=yt,n=St,y=null,b=T(0),l=T(G),p=T(0);function i(e){var r,s=(e=rt(e)).length,h,w,$=0,f=new Array(s),o=new Array(s),D=+b.apply(this,arguments),z=Math.min(G,Math.max(-G,l.apply(this,arguments)-D)),k,R=Math.min(Math.abs(z)/s,p.apply(this,arguments)),u=R*(z<0?-1:1),A;for(r=0;r<s;++r)(A=o[f[r]=r]=+t(e[r],r,e))>0&&($+=A);for(n!=null?f.sort(function(E,m){return n(o[E],o[m])}):y!=null&&f.sort(function(E,m){return y(e[E],e[m])}),r=0,w=$?(z-s*u)/$:0;r<s;++r,D=k)h=f[r],A=o[h],k=D+(A>0?A*w:0)+u,o[h]={data:e[h],index:r,value:A,startAngle:D,endAngle:k,padAngle:R};return o}return i.value=function(e){return arguments.length?(t=typeof e=="function"?e:T(+e),i):t},i.sortValues=function(e){return arguments.length?(n=e,y=null,i):n},i.sort=function(e){return arguments.length?(y=e,n=null,i):y},i.startAngle=function(e){return arguments.length?(b=typeof e=="function"?e:T(+e),i):b},i.endAngle=function(e){return arguments.length?(l=typeof e=="function"?e:T(+e),i):l},i.padAngle=function(e){return arguments.length?(p=typeof e=="function"?e:T(+e),i):p},i}var At=mt.pie,I={sections:new Map,showData:!1},_=I.sections,V=I.showData,Ct=structuredClone(At),$t=g(()=>structuredClone(Ct),"getConfig"),Dt=g(()=>{_=new Map,V=I.showData,ft()},"clear"),Tt=g(({label:t,value:n})=>{if(n<0)throw new Error(`"${t}" has invalid value: ${n}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);_.has(t)||(_.set(t,n),B.debug(`added new section: ${t}, with value: ${n}`))},"addSection"),bt=g(()=>_,"getSections"),kt=g(t=>{V=t},"setShowData"),Mt=g(()=>V,"getShowData"),J={getConfig:$t,clear:Dt,setDiagramTitle:ct,getDiagramTitle:lt,setAccTitle:st,getAccTitle:ot,setAccDescription:it,getAccDescription:nt,addSection:Tt,getSections:bt,setShowData:kt,getShowData:Mt},zt=g((t,n)=>{at(t,n),n.setShowData(t.showData),t.sections.map(n.addSection)},"populateDb"),Et={parse:g(async t=>{const n=await vt("pie",t);B.debug(n),zt(n,J)},"parse")},Lt=g(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieCircle.highlighted{
    scale: 1.05;
    opacity: 1;
  }
  .pieCircle.highlightedOnHover:hover{
    transition-duration: 250ms;
    scale: 1.05;
    opacity: 1;
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),Rt=Lt,Ot=g(t=>{const n=[...t.values()].reduce((l,p)=>l+p,0),y=[...t.entries()].map(([l,p])=>({label:l,value:p})).filter(l=>l.value/n*100>=1);return wt().value(l=>l.value).sort(null)(y)},"createPieArcs"),Wt=g((t,n,y,b)=>{B.debug(`rendering pie chart
`+t);const l=b.db,p=ut(),i=dt(l.getConfig(),p.pie),e=40,r=18,s=4,h=450,w=h,$=gt(n),f=$.append("g");f.attr("transform","translate("+w/2+","+h/2+")");const{themeVariables:o}=p;let[D]=pt(o.pieOuterStrokeWidth);D??=2;const z=i.legendPosition,k=i.textPosition,R=i.donutHole>0&&i.donutHole<=.9?i.donutHole:0,u=Math.min(w,h)/2-e,A=q().innerRadius(R*u).outerRadius(u),E=q().innerRadius(u*k).outerRadius(u*k),m=f.append("g");m.append("circle").attr("cx",0).attr("cy",0).attr("r",u+D/2).attr("class","pieOuterCircle");const O=l.getSections(),K=Ot(O),Q=[o.pie1,o.pie2,o.pie3,o.pie4,o.pie5,o.pie6,o.pie7,o.pie8,o.pie9,o.pie10,o.pie11,o.pie12];let F=0;O.forEach(a=>{F+=a});const j=K.filter(a=>(a.data.value/F*100).toFixed(0)!=="0"),H=xt(Q).domain([...O.keys()]);m.selectAll("mySlices").data(j).enter().append("path").attr("d",A).attr("fill",a=>H(a.data.label)).attr("class",a=>{let c="pieCircle";return i.highlightSlice==="hover"?c+=" highlightedOnHover":i.highlightSlice===a.data.label&&(c+=" highlighted"),c}),m.selectAll("mySlices").data(j).enter().append("text").text(a=>(a.data.value/F*100).toFixed(0)+"%").attr("transform",a=>"translate("+E.centroid(a)+")").style("text-anchor","middle").attr("class","slice");const Y=f.append("text").text(l.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText"),L=[...O.entries()].map(([a,c])=>({label:a,value:c})),C=f.selectAll(".legend").data(L).enter().append("g").attr("class","legend");C.append("rect").attr("width",r).attr("height",r).style("fill",a=>H(a.label)).style("stroke",a=>H(a.label)),C.append("text").attr("x",r+s).attr("y",r-s).text(a=>l.getShowData()?`${a.label} [${a.value}]`:a.label);const M=Math.max(...C.selectAll("text").nodes().map(a=>a?.getBoundingClientRect().width??0));let W=h,N=w+e;const d=r+s,P=L.length*d;switch(z){case"center":C.attr("transform",(a,c)=>{const v=d*L.length/2,x=-M/2-(r+s),S=c*d-v;return"translate("+x+","+S+")"});break;case"top":W+=P,C.attr("transform",(a,c)=>{const v=u,x=-M/2-(r+s),S=c*d-v;return`translate(${x}, ${S})`}),m.attr("transform",()=>`translate(0, ${P+d})`);break;case"bottom":W+=P,C.attr("transform",(a,c)=>{const v=-u-d,x=-M/2-(r+s),S=c*d-v;return"translate("+x+","+S+")"});break;case"left":N+=r+s+M,C.attr("transform",(a,c)=>{const v=d*L.length/2,x=-u-(r+s),S=c*d-v;return"translate("+x+","+S+")"}),m.attr("transform",()=>`translate(${M+r+s}, 0)`);break;default:N+=r+s+M,C.attr("transform",(a,c)=>{const v=d*L.length/2,x=12*r,S=c*d-v;return"translate("+x+","+S+")"});break}const U=Y.node()?.getBoundingClientRect().width??0,tt=w/2-U/2,et=w/2+U/2,X=Math.min(0,tt),Z=Math.max(N,et)-X;$.attr("viewBox",`${X} 0 ${Z} ${W}`),ht($,W,Z,i.useMaxWidth)},"draw"),_t={draw:Wt},It={parser:Et,db:J,renderer:_t,styles:Rt};export{It as diagram};
