"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Client, handle_file } from "@gradio/client";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Loader2, Search, Database, Cpu, Target, 
  BarChart3, RefreshCw, Layers, CheckCircle2,
  Car, Bird, Package, Clipboard, Info, ArrowRight
} from "lucide-react";

// --- Types ---
interface GradioImageData { url?: string; path?: string; }
interface GradioImage { image?: GradioImageData; url?: string; path?: string; caption?: string; }
interface ParsedStats {
  precision: number; mapR: number;
  recall: { r1: number; r5: number; r10: number; r100: number };
  embeddings: string; dim: number;
}

type LoadResourcesResponse = [string, string];
type RefreshExamplesResponse = [GradioImage[]];
type SearchResponse = [GradioImage[], string];

export default function DinoRetrievalApp() {
  const [dataset, setDataset] = useState<string>("Cars196");
  const [size, setSize] = useState<string>("b"); 
  const [isFinetuned, setIsFinetuned] = useState<boolean>(false);
  const [topK, setTopK] = useState<number>(10);

  const [status, setStatus] = useState<string>("Initializing...");
  const [stats, setStats] = useState<ParsedStats | null>(null);
  const [examples, setExamples] = useState<GradioImage[]>([]);
  const [results, setResults] = useState<GradioImage[]>([]);
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"example" | "upload">("example");

  const clientRef = useRef<Client | null>(null);

  const getImageUrl = (item: GradioImage): string => {
    if (!item) return "";
    return item.image?.url || item.image?.path || item.url || item.path || "";
  };

  const parseStats = (raw: string): ParsedStats | null => {
    try {
      const getNum = (regex: RegExp) => {
        const match = raw.match(regex);
        return match ? parseFloat(match[1]) : 0;
      };
      return {
        precision: getNum(/Precision@1: ([\d.]+)%/),
        mapR: getNum(/MAP@R: ([\d.]+)%/),
        recall: {
          r1: getNum(/R@1: ([\d.]+)%/), r5: getNum(/R@5: ([\d.]+)%/),
          r10: getNum(/R@10: ([\d.]+)%/), r100: getNum(/R@100: ([\d.]+)%/),
        },
        embeddings: raw.match(/Total embeddings: (\d+)/)?.[1] || "0",
        dim: parseInt(raw.match(/Embedding dimension: (\d+)/)?.[1] || "0"),
      };
    } catch { return null; }
  };

  const getGradioClient = async (): Promise<Client> => {
    if (clientRef.current) return clientRef.current;
    const client = await Client.connect("pawlo2013/Dinov3_Image_Retrieval");
    clientRef.current = client;
    return client;
  };

  const handleSearch = useCallback(async (source: GradioImage | File | Blob) => {
    setIsSearching(true);
    setStatus("Analyzing Visual Features...");
    
    let previewUrl = "";
    if (source instanceof File || source instanceof Blob) {
      previewUrl = URL.createObjectURL(source);
    } else {
      previewUrl = getImageUrl(source);
    }
    setActiveQuery(previewUrl);

    try {
      const client = await getGradioClient();
      const input = (source instanceof File || source instanceof Blob) 
        ? source 
        : handle_file(getImageUrl(source));
      
      const result = await client.predict("/process_image", {
        image_input: input,
        k_neighbors: topK,
      });

      const data = result.data as SearchResponse;
      setResults(data[0]);
      setStatus(data[1]);
    } catch (err) {
      console.error(err);
      setStatus("Search Failed");
    } finally {
      setIsSearching(false);
    }
  }, [topK]);

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const blob = items[i].getAsFile();
          if (blob) handleSearch(blob);
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [handleSearch]);

  const loadResources = async () => {
    setIsLoading(true);
    setStatus("Loading Model & Index...");
    try {
      const client = await getGradioClient();
      const result = await client.predict("/load_resources", {
        dataset, dino_version: "3", dino_size: size, is_finetuned: isFinetuned,
      });
      const data = result.data as LoadResourcesResponse;
      setStatus(data[0]);
      setStats(parseStats(data[1]));
      
      const exResult = await client.predict("/refresh_examples_wrapper", {});
      const exData = exResult.data as RefreshExamplesResponse;
      setExamples(exData[0] || []);
    } catch { setStatus("Sync Failed"); } finally { setIsLoading(false); }
  };

  useEffect(() => { loadResources(); }, [dataset, size, isFinetuned]);

  const datasets = [
    { id: "Cars196", name: "Cars-196", icon: <Car size={20} />, desc: "196 Car classes" },
    { id: "CUB", name: "CUB-200", icon: <Bird size={20} />, desc: "200 Bird species" },
    { id: "StanfordOnlineProducts", name: "Products", icon: <Package size={20} />, desc: "Online products" },
  ];

  return (
    <div className="min-h-screen bg-[#f1f5f9] text-slate-900 font-sans pb-20">
      {isLoading && <div className="fixed top-0 left-0 right-0 h-1 bg-blue-600 z-50 animate-pulse" />}

      <div className="max-w-7xl mx-auto p-4 md:p-8 space-y-8">
        {/* Header & Fixed Info Box */}
        <header className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
          <div className="space-y-4">
            <motion.h1 initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-4xl font-black tracking-tighter text-slate-900">
              🦖 DINO<span className="text-blue-600">v3</span> Retrieval
            </motion.h1>
            
            {/* FIXED INFO BOX LAYOUT */}
            <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
              <div className="flex gap-4">
                <div className="bg-blue-50 p-2 rounded-full h-fit">
                  <Info className="text-blue-500" size={20} />
                </div>
                <div className="text-sm text-slate-600 leading-relaxed">
                  <p>
                    This system uses <strong className="font-bold text-slate-900">DINOv3 </strong>, a state-of-the-art foundation vision model. 
                    <br />
                    For more info about Dinov3 see the <a href="https://ai.meta.com/dinov3/" target="_blank" rel="noreferrer" className="text-blue-600 underline font-medium">site. </a> In this demo you can select from different image retrieval datasets to explore how well DINOv3 performs in finding visually similar images.
                  </p>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-4 pt-2 border-t border-slate-50">
                {[
                  { label: "SELECT EXAMPLE", icon: <CheckCircle2 size={14} /> },
                  { label: "UPLOAD IMAGE", icon: <CheckCircle2 size={14} /> },
                  { label: "PASTE (CTRL+V)", icon: <CheckCircle2 size={14} /> }
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-1.5 text-[10px] font-black uppercase tracking-widest text-slate-400">
                    <span className="text-blue-500">{item.icon}</span>
                    {item.label}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="flex flex-col md:items-end gap-2">
            <div className="flex items-center gap-3 bg-white p-3 rounded-2xl border shadow-sm px-6">
              <div className={`w-2.5 h-2.5 rounded-full ${isLoading || isSearching ? 'bg-amber-400 animate-pulse' : 'bg-emerald-500'}`} />
              <span className="text-xs font-mono font-black text-slate-700 uppercase tracking-tight">{status}</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-4 space-y-6">
            <section className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200 space-y-6">
              <div className="space-y-2">
                <h3 className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                  <Database size={14} /> Domain Dataset
                </h3>
                <div className="grid grid-cols-1 gap-2 pt-1">
                  {datasets.map((d) => (
                    <button key={d.id} onClick={() => setDataset(d.id)} className={`flex items-center gap-4 p-3 rounded-2xl border-2 transition-all text-left ${dataset === d.id ? 'border-blue-600 bg-blue-50 shadow-md' : 'border-slate-100 hover:border-slate-200'}`}>
                      <div className={`p-2.5 rounded-xl transition-colors ${dataset === d.id ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-400'}`}>{d.icon}</div>
                      <div>
                        <p className="font-bold text-sm text-slate-800">{d.name}</p>
                        <p className="text-[10px] text-slate-400 font-medium uppercase tracking-tighter">{d.desc}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Scale</label>
                  <div className="flex p-1 bg-slate-100 rounded-xl">
                    {['s', 'b'].map(v => (
                      <button key={v} onClick={() => setSize(v)} className={`flex-1 py-1.5 text-xs font-bold rounded-lg transition-all ${size === v ? 'bg-white shadow-sm text-blue-600' : 'text-slate-400 hover:text-slate-500'}`}>
                        {v === 's' ? 'Small' : 'Base'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Weights</label>
                  <button onClick={() => setIsFinetuned(!isFinetuned)} className={`w-full py-2.5 rounded-xl border text-[10px] font-black uppercase tracking-tighter transition-all ${isFinetuned ? 'bg-blue-600 text-white border-blue-600 shadow-sm' : 'bg-slate-50 text-slate-400 border-slate-200'}`}>
                    {isFinetuned ? 'Finetuned' : 'Pretrained'}
                  </button>
                </div>
              </div>

              <div className="space-y-3 pt-2">
                <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-widest">
                  <span>Retrieve K</span>
                  <span className="text-blue-600 font-mono text-xs">{topK}</span>
                </div>
                <input type="range" min="1" max="50" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600" />
              </div>
            </section>

            <AnimatePresence>
              {stats && (
                <motion.section initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-gray-500 p-6 rounded-3xl shadow-xl text-white space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/10 p-4 rounded-2xl border border-white/5">
                      <p className="text-[9px] font-black text-white/40 uppercase tracking-widest">Precision@1</p>
                      <p className="text-2xl font-black tracking-tight">{stats.precision}%</p>
                    </div>
                    <div className="bg-white/10 p-4 rounded-2xl border border-white/5">
                      <p className="text-[9px] font-black text-white/40 uppercase tracking-widest">Recall@1</p>
                      <p className="text-2xl font-black tracking-tight">{stats.recall.r1}%</p>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-[10px] font-black text-white/30 uppercase tracking-widest">
                      <span className="flex items-center gap-2"><Target size={12}/> Global MAP@R</span>
                      <span>{stats.mapR}%</span>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${stats.mapR}%` }} className="h-full bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)]" />
                    </div>
                  </div>
                </motion.section>
              )}
            </AnimatePresence>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-8 space-y-6">
            <AnimatePresence>
              {activeQuery && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white p-5 rounded-3xl border border-blue-100 shadow-md flex items-center gap-6">
                  <div className="relative shrink-0">
                    <img src={activeQuery} className="w-20 h-20 rounded-2xl object-cover shadow-inner bg-slate-50 border border-slate-100" />
                    {isSearching && (
                      <div className="absolute inset-0 bg-blue-600/30 backdrop-blur-[1px] animate-pulse flex items-center justify-center rounded-2xl">
                        <Loader2 className="animate-spin text-white" size={24} />
                      </div>
                    )}
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-blue-600 font-black text-[10px] uppercase tracking-widest">
                      <Search size={14}/> Query Vectorized
                    </div>
                    <p className="text-slate-900 font-bold text-lg leading-tight">Retrieving Visual Similars</p>
                    <p className="text-slate-400 text-xs font-medium italic">Index: {dataset} • Dim: 1024</p>
                  </div>
                  <div className="ml-auto opacity-10 hidden md:block">
                     <ArrowRight size={48} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="flex p-1.5 bg-slate-50 m-2 rounded-2xl border">
                {(["example", "upload"] as const).map((tab) => (
                  <button key={tab} onClick={() => setActiveTab(tab)} className={`flex-1 py-2.5 text-xs font-black uppercase tracking-widest rounded-xl transition-all ${activeTab === tab ? 'bg-white shadow-md text-blue-600' : 'text-slate-400 hover:text-slate-600'}`}>
                    {tab === 'example' ? 'Select Test' : 'Custom Upload'}
                  </button>
                ))}
              </div>

              <div className="p-6 pt-2">
                {activeTab === "example" ? (
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                    {examples.map((item, idx) => (
                      <motion.img 
                        key={idx} 
                        whileHover={{ scale: 1.05, zIndex: 10 }} 
                        whileTap={{ scale: 0.95 }} 
                        src={getImageUrl(item)} 
                        className="w-full aspect-square object-cover rounded-2xl cursor-pointer shadow-sm border border-transparent hover:border-blue-400 transition-all bg-slate-50" 
                        onClick={() => handleSearch(item)} 
                      />
                    ))}
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-slate-200 rounded-3xl p-10 text-center space-y-4 hover:border-blue-400 transition-all bg-slate-50 group">
                    <div className="flex flex-wrap justify-center gap-4">
                      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex flex-col items-center gap-2 transition-transform group-hover:translate-y-[-2px]">
                        <Clipboard className="text-blue-500" size={24} />
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Paste Image</span>
                      </div>
                      <label className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex flex-col items-center gap-2 cursor-pointer transition-transform group-hover:translate-y-[-2px]">
                        <Search className="text-blue-500" size={24} />
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Browse File</span>
                        <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && handleSearch(e.target.files[0])} className="hidden" />
                      </label>
                    </div>
                    <p className="text-[11px] font-medium text-slate-400 italic">Drag & drop images here or use keyboard shortcuts</p>
                  </div>
                )}
              </div>
            </div>

            <section className="space-y-4">
              <h3 className="text-xl font-black text-slate-900 flex items-center gap-2 px-2">Matches <span className="text-sm font-medium text-slate-300">({results.length})</span></h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-4">
                {results.length > 0 ? results.map((res, idx) => (
                  <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: idx * 0.03 }} key={idx} className="bg-white p-2 rounded-2xl shadow-sm border border-slate-100 group">
                    <img src={getImageUrl(res)} className="w-full aspect-square object-cover rounded-xl group-hover:scale-[1.03] transition-transform bg-slate-50 shadow-inner" />
                    {res.caption && <p className="text-[9px] mt-2 text-slate-400 font-black truncate text-center uppercase tracking-tighter px-1">{res.caption}</p>}
                  </motion.div>
                )) : (
                  <div className="col-span-full py-32 text-center text-slate-300 flex flex-col items-center gap-4 bg-white rounded-3xl border border-dashed border-slate-200">
                    <div className="bg-slate-50 p-4 rounded-full">
                      <Search size={40} strokeWidth={1.5} className="text-slate-200" />
                    </div>
                    <p className="font-bold text-slate-400 uppercase tracking-widest text-xs">Awaiting Query Input</p>
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}