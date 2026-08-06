#!/usr/bin/env python3
"""Run the concise HyperSense SWAN wearable analysis.

Outputs: three paper figures, three main tables, two supplement tables, captions,
and a short results summary. All associations are participant-level Spearman
correlations and exploratory (n=12).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STUDY_DATES = {
    "P001": {"2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07"}, "P002": {"2025-02-03","2025-02-04","2025-02-05"},
    "P003": {"2025-02-03","2025-02-04"}, "P004": {"2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14"},
    "P005": {"2025-02-10","2025-02-11","2025-02-12"}, "P006": {"2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28"},
    "P007": {"2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28"}, "P008": {"2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28"},
    "P009": {"2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07"}, "P012": {"2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07"},
    "P014": {"2025-03-25","2025-03-26","2025-03-27","2025-03-31","2025-04-01"}, "P016": {"2025-03-25","2025-03-26","2025-03-27","2025-03-31","2025-04-01"},
}
SCORES = ["teacher_inattention_swan","teacher_hyperactivity_swan","teacher_overall_swan","parent_inattention_swan","parent_hyperactivity_swan","parent_overall_swan"]

def bh(p: pd.Series) -> pd.Series:
    out=pd.Series(np.nan,index=p.index,dtype=float); ok=p.notna(); v=p[ok].to_numpy(float); order=np.argsort(v)
    adj=np.minimum.accumulate((v[order]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]; restore=np.empty_like(adj); restore[order]=np.minimum(adj,1); out.loc[ok]=restore; return out

def ci(x,y,rng,n=2000):
    values=[]
    for _ in range(n):
        ix=rng.integers(0,len(x),len(x)); r=spearmanr(x[ix],y[ix]).statistic
        if np.isfinite(r): values.append(r)
    return (np.quantile(values,.025),np.quantile(values,.975)) if values else (np.nan,np.nan)

def read_watch_metrics(path: Path,pid: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Return valid study-period heart rate and Apple-Watch step records.

    Health exports do not provide raw accelerometer samples. StepCount from an
    Apple Watch source is therefore retained as the Watch movement proxy.
    """
    with path.open(encoding="utf-8-sig",errors="ignore") as h: header=next(i for i,line in enumerate(h) if "/@locale" in line)
    raw=pd.read_csv(path,skiprows=header,usecols=["/Record/@startDate","/Record/@type","/Record/@value","/Record/@sourceName"],low_memory=False)
    raw["ts"]=pd.to_datetime(raw["/Record/@startDate"],errors="coerce",utc=True).dt.tz_convert("US/Pacific"); raw["hr"]=pd.to_numeric(raw["/Record/@value"],errors="coerce")
    raw=raw.dropna(subset=["ts","hr"]); raw["date"]=raw.ts.dt.strftime("%Y-%m-%d")
    raw=raw[raw.date.isin(STUDY_DATES[pid])].copy()
    hr=raw[raw["/Record/@type"].eq("HKQuantityTypeIdentifierHeartRate") & raw.hr.between(40,180)].copy()
    source=raw["/Record/@sourceName"].astype(str).str.replace("\u00a0"," ",regex=False)
    steps=raw[raw["/Record/@type"].eq("HKQuantityTypeIdentifierStepCount") & source.str.contains("Apple Watch",case=False,na=False,regex=False) & raw.hr.ge(0)].copy()
    return hr,steps

def load_swan(path,crosswalk):
    source=pd.read_excel(path); name="Child Name / ID\n"; needed={name,"Inattention (1-9)","Hyperactivity (10-18)"}
    if needed-set(source): raise ValueError("SWAN workbook columns do not match the expected export.")
    x=source[[name,"Inattention (1-9)","Hyperactivity (10-18)"]].rename(columns={name:"source","Inattention (1-9)":"inattention","Hyperactivity (10-18)":"hyperactivity"})
    x.source=x.source.astype(str).str.strip(); x.inattention=pd.to_numeric(x.inattention,errors="coerce"); x.hyperactivity=pd.to_numeric(x.hyperactivity,errors="coerce"); x=x.dropna(); x["overall"]=(x.inattention+x.hyperactivity)/2
    rows=[]
    for _,r in crosswalk.iterrows():
        row={"participant_id":r.participant_id,"child_label":r.child_label}
        for rater,column in [("parent","parent_source_label"),("teacher","teacher_source_label")]:
            match=x[x.source.eq(r[column].strip())]
            if len(match)!=1: raise ValueError(f"Expected exactly one SWAN match for {r.participant_id} {rater}.")
            for d in ["inattention","hyperactivity","overall"]: row[f"{rater}_{d}_swan"]=match.iloc[0][d]
        rows.append(row)
    return pd.DataFrame(rows)

def load_mocopi(root: Path):
    files=sorted(set(root.glob("P*_epoch_kinematics.csv"))|set(root.glob("P*/P*_epoch_kinematics.csv")))
    if not files: raise FileNotFoundError("No P###_epoch_kinematics.csv files found.")
    frames=[]
    for f in files:
        d=pd.read_csv(f,low_memory=False); req={"Participant","Sensor","Intensity","Jerk","class"}
        if req-set(d): raise ValueError(f"{f.name} is missing required epoch columns.")
        for c in ["Intensity","Jerk"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        frames.append(d.dropna(subset=["Participant","Sensor","Intensity","Jerk"]))
    return pd.concat(frames,ignore_index=True)

def corr_table(data, features, scores, family, rng):
    rows=[]
    for feature,label in features:
        for score in scores:
            z=data[[feature,score]].dropna(); rho,p=spearmanr(z[feature],z[score]); lo,hi=ci(z[feature].to_numpy(),z[score].to_numpy(),rng)
            rater,domain,_=score.split("_",2); rows.append({"analysis_family":family,"wearable_measure":label,"feature":feature,"rater":rater,"swan_domain":domain,"n":len(z),"spearman_rho":rho,"ci_95_low":lo,"ci_95_high":hi,"p_value":p})
    out=pd.DataFrame(rows); out["p_value_fdr_bh"]=bh(out.p_value); return out

def scatter(ax,data,x,y,xlabel,ylabel,title):
    # CSV caches may reload numeric values with an ``object`` dtype.  Coerce
    # explicitly here so NumPy's regression and Matplotlib receive floats.
    z=data[[x,y,"participant_id"]].copy()
    z[x]=pd.to_numeric(z[x],errors="coerce"); z[y]=pd.to_numeric(z[y],errors="coerce")
    z=z.dropna(subset=[x,y]); r,p=spearmanr(z[x],z[y]); ax.scatter(z[x],z[y],color="#1967d2",s=52)
    m,b=np.polyfit(z[x].to_numpy(float),z[y].to_numpy(float),1); grid=np.linspace(z[x].min(),z[x].max(),100); ax.plot(grid,m*grid+b,color="#d93025",lw=1.5)
    for _,row in z.iterrows(): ax.annotate(row.participant_id,(row[x],row[y]),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f"{title}\nrho={r:.2f}, p={p:.3f}, n={len(z)}"); ax.grid(alpha=.2)

def figures(data,head,out):
    fig,ax=plt.subplots(figsize=(6.4,5)); scatter(ax,data,"mean_hr_bpm","teacher_overall_swan","Mean heart rate (bpm)","Teacher overall SWAN","Apple Watch heart rate and teacher overall SWAN"); fig.tight_layout(); fig.savefig(out/"figure_1_apple_watch_heart_rate.png",dpi=300); plt.close(fig)
    pairs=[("mean_accel","teacher_inattention_swan","Head acceleration","Teacher inattention SWAN"),("mean_jerk","teacher_inattention_swan","Head jerk","Teacher inattention SWAN"),("mean_accel","teacher_overall_swan","Head acceleration","Teacher overall SWAN"),("mean_jerk","teacher_overall_swan","Head jerk","Teacher overall SWAN")]
    fig,axes=plt.subplots(2,2,figsize=(11,8.4))
    for ax,(x,y,xl,yl) in zip(axes.ravel(),pairs): scatter(ax,head,x,y,xl,yl,"Head-sensor movement and teacher SWAN")
    fig.suptitle("Head-sensor MOCOPI associations with retained teacher SWAN ratings",fontsize=15); fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out/"figure_2_head_sensor_mocopi.png",dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    for ax,d in zip(axes,["inattention","hyperactivity","overall"]):
        x,y=f"parent_{d}_swan",f"teacher_{d}_swan"; r,p=spearmanr(data[x],data[y]); ax.scatter(data[x],data[y],color="#1967d2",s=45); low=min(data[x].min(),data[y].min()); high=max(data[x].max(),data[y].max()); ax.plot([low,high],[low,high],"--",color="gray",lw=1)
        for _,row in data.iterrows(): ax.annotate(row.participant_id,(row[x],row[y]),xytext=(3,3),textcoords="offset points",fontsize=7)
        ax.set_xlabel(f"Parent {d.title()} SWAN"); ax.set_ylabel(f"Teacher {d.title()} SWAN"); ax.set_title(f"{d.title()}\nrho={r:.2f}, p={p:.3f}"); ax.grid(alpha=.2)
    fig.suptitle("Parent and retained-teacher SWAN ratings",fontsize=15); fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(out/"figure_3_parent_teacher_agreement.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,5)); scatter(ax,data,"mean_daily_watch_steps","teacher_inattention_swan","Mean daily Apple Watch steps","Teacher inattention SWAN","Apple Watch movement proxy and teacher inattention SWAN"); fig.tight_layout(); fig.savefig(out/"figure_7_apple_watch_steps.png",dpi=300); plt.close(fig)

def sensor_profile_figure(sensor, out, rng):
    """One compact robustness check: acceleration associations across sensors."""
    sensors=sorted(sensor.Sensor.dropna().unique())
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
    for ax,score,title in zip(
        axes,
        ["teacher_inattention_swan","teacher_overall_swan"],
        ["Teacher inattention SWAN","Teacher overall SWAN"],
    ):
        rows=[]
        for name in sensors:
            z=sensor.loc[sensor.Sensor.eq(name),["mean_accel",score]].dropna()
            rho,p=spearmanr(z["mean_accel"],z[score])
            lo,hi=ci(z["mean_accel"].to_numpy(float),z[score].to_numpy(float),rng)
            rows.append((name,rho,lo,hi,p))
        # Put Head at the top, then retain an alphabetical, reproducible order.
        rows.sort(key=lambda row: (row[0] != "Head", row[0]))
        y=np.arange(len(rows))
        colors=["#d93025" if name=="Head" else "#1967d2" for name,*_ in rows]
        for yy,(_,rho,lo,hi,_),color in zip(y,rows,colors):
            ax.plot([lo,hi],[yy,yy],color=color,lw=2)
            ax.scatter(rho,yy,color=color,s=48,zorder=3)
        ax.axvline(0,color="gray",lw=1,ls="--")
        ax.set_yticks(y,[name for name,*_ in rows])
        ax.invert_yaxis(); ax.set_xlim(-1,1); ax.set_xlabel("Spearman correlation with mean acceleration")
        ax.set_title(title); ax.grid(axis="x",alpha=.2)
    fig.suptitle("Teacher-SWAN association with mean acceleration, by MOCOPI sensor\n(points: rho; lines: 95% bootstrap CI; Head highlighted)",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(out/"figure_4_sensor_acceleration_profile.png",dpi=300); plt.close(fig)

def coverage_figure(data, out):
    """Document analytic exposure without adding another outcome comparison."""
    z=data.sort_values("participant_id")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,column,title in zip(
        axes,
        ["valid_hr_records","valid_watch_step_records","valid_epochs"],
        ["Valid Apple Watch heart-rate records","Valid Apple Watch step records","Valid MOCOPI one-minute epochs"],
    ):
        ax.barh(z.participant_id,z[column],color="#1967d2")
        ax.set_xscale("log"); ax.set_xlabel("Number of valid records (log scale)"); ax.set_title(title); ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Wearable data coverage by participant",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(out/"figure_5_data_coverage.png",dpi=300); plt.close(fig)

def class_context_figure(cls, out, rng):
    """Exploratory classroom check, restricted to contexts with >=5 children."""
    groups=[(name,g) for name,g in cls.groupby("class") if len(g)>=5]
    groups.sort(key=lambda item: item[0])
    fig,axes=plt.subplots(1,2,figsize=(11,5.1),sharey=True)
    for ax,score,title in zip(axes,["teacher_inattention_swan","teacher_overall_swan"],["Teacher inattention SWAN","Teacher overall SWAN"]):
        rows=[]
        for name,g in groups:
            z=g[["mean_accel",score]].dropna(); rho,p=spearmanr(z["mean_accel"],z[score]); lo,hi=ci(z["mean_accel"].to_numpy(float),z[score].to_numpy(float),rng)
            rows.append((name,len(z),rho,lo,hi))
        y=np.arange(len(rows))
        for yy,(_,_,rho,lo,hi) in zip(y,rows):
            ax.plot([lo,hi],[yy,yy],color="#1967d2",lw=2); ax.scatter(rho,yy,color="#1967d2",s=48,zorder=3)
        ax.axvline(0,color="gray",lw=1,ls="--"); ax.set_xlim(-1,1)
        ax.set_yticks(y,[f"{name} (n={n})" for name,n,*_ in rows]); ax.invert_yaxis()
        ax.set_xlabel("Spearman correlation with head acceleration"); ax.set_title(title); ax.grid(axis="x",alpha=.2)
    fig.suptitle("Exploratory head-acceleration association by classroom context\n(points: rho; lines: 95% bootstrap CI; contexts with n≥5 only)",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(out/"figure_6_classroom_context.png",dpi=300); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--apple-root",required=True,type=Path); p.add_argument("--swan-workbook",required=True,type=Path); p.add_argument("--mocopi-root",required=True,type=Path); p.add_argument("--crosswalk",type=Path,default=Path("SWAN/inputs/participant_crosswalk.csv")); p.add_argument("--output-dir",type=Path,default=Path("SWAN/outputs")); a=p.parse_args()
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); cross=pd.read_csv(a.crosswalk,dtype=str).fillna(""); cross=cross[cross.include_in_analysis.str.lower().eq("yes")]
    swan=load_swan(a.swan_workbook,cross); mocopi=load_mocopi(a.mocopi_root); pids=cross.participant_id.tolist(); rng=np.random.default_rng(20260805)
    cache_path=out/".apple_hr_cache.csv"; apple_columns=["participant_id","valid_hr_records","mean_hr_bpm","sd_hr_bpm","valid_watch_step_records","mean_daily_watch_steps","metric_version"]
    apple=pd.read_csv(cache_path) if cache_path.is_file() else pd.DataFrame(columns=apple_columns)
    if set(apple_columns)-set(apple.columns) or not apple.metric_version.eq(4).all(): apple=pd.DataFrame(columns=apple_columns)
    apple=apple[apple.participant_id.isin(pids)].copy()
    for pid in pids:
        if pid in set(apple.participant_id): continue
        print(f"[INFO] Reading Apple Watch {pid}",flush=True); h,steps=read_watch_metrics(a.apple_root/f"{pid}export.csv",pid)
        apple=pd.concat([apple,pd.DataFrame([{"participant_id":pid,"valid_hr_records":len(h),"mean_hr_bpm":h.hr.mean(),"sd_hr_bpm":h.hr.std(ddof=1),"valid_watch_step_records":len(steps),"mean_daily_watch_steps":steps.hr.sum()/len(STUDY_DATES[pid]) if len(steps) else np.nan,"metric_version":4}])],ignore_index=True).drop_duplicates("participant_id",keep="last")
        apple.to_csv(cache_path,index=False)
    apple=apple.sort_values("participant_id").drop(columns="metric_version"); total=mocopi.groupby("Participant",as_index=False).agg(valid_epochs=("Intensity","size"),mean_accel=("Intensity","mean"),mean_jerk=("Jerk","mean"),sensor_count=("Sensor","nunique")).rename(columns={"Participant":"participant_id"})
    data=swan.merge(apple,on="participant_id",validate="one_to_one").merge(total,on="participant_id",validate="one_to_one"); data.to_csv(out/"analysis_dataset.csv",index=False)
    data[["participant_id","valid_hr_records","valid_watch_step_records","valid_epochs","sensor_count"]].to_csv(out/"table_1_participant_coverage.csv",index=False)
    primary=pd.concat([corr_table(data,[("mean_hr_bpm","Mean all-study-period heart rate")],SCORES,"Apple Watch heart rate",rng),corr_table(data,[("mean_daily_watch_steps","Mean daily Apple Watch steps")],SCORES,"Apple Watch movement proxy",rng),corr_table(data,[("mean_accel","Mean all-sensor acceleration"),("mean_jerk","Mean all-sensor jerk")],SCORES,"Total MOCOPI movement",rng)],ignore_index=True); primary.to_csv(out/"table_2_primary_correlations.csv",index=False)
    sensor=mocopi.groupby(["Participant","Sensor"],as_index=False).agg(valid_epochs=("Intensity","size"),mean_accel=("Intensity","mean"),mean_jerk=("Jerk","mean")).merge(swan,left_on="Participant",right_on="participant_id",how="inner"); head=sensor[sensor.Sensor.eq("Head")].copy(); head_results=corr_table(head,[("mean_accel","Head acceleration"),("mean_jerk","Head jerk")],["teacher_inattention_swan","teacher_overall_swan"],"Head sensor replication",rng); head_results.to_csv(out/"table_3_head_sensor_replication.csv",index=False)
    all_sensor=[]
    for s in sorted(sensor.Sensor.unique()): all_sensor.append(corr_table(sensor[sensor.Sensor.eq(s)],[("mean_accel","Acceleration"),("mean_jerk","Jerk")],SCORES[:3],"All sensors (teacher)",rng).assign(sensor=s))
    pd.concat(all_sensor,ignore_index=True).to_csv(out/"supplement_table_s1_all_sensor_correlations.csv",index=False)
    cls=mocopi[mocopi.Sensor.eq("Head")].groupby(["Participant","class"],as_index=False).agg(mean_accel=("Intensity","mean"),mean_jerk=("Jerk","mean")).merge(swan,left_on="Participant",right_on="participant_id",how="inner"); class_rows=[]
    for c,g in cls.groupby("class"):
        for f in ["mean_accel","mean_jerk"]:
            for score in ["teacher_inattention_swan","teacher_overall_swan"]:
                z=g[[f,score]].dropna(); r,pv=(spearmanr(z[f],z[score]) if len(z)>=5 else (np.nan,np.nan)); class_rows.append({"class_label":c,"feature":f,"swan_outcome":score,"n":len(z),"spearman_rho":r,"p_value":pv})
    class_table=pd.DataFrame(class_rows); class_table["p_value_fdr_bh"]=bh(class_table.p_value); class_table.to_csv(out/"supplement_table_s2_head_sensor_class_correlations.csv",index=False)
    figures(data,head,out); sensor_profile_figure(sensor,out,rng); coverage_figure(data,out); class_context_figure(cls,out,rng)
    cache_path.unlink(missing_ok=True)
    print(f"Wrote concise analysis outputs to {out}")

if __name__=="__main__": main()
