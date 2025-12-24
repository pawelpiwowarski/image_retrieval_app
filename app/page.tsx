"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Client, handle_file } from "@gradio/client";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Loader2, Search, Database, Cpu, Target, 
  BarChart3, RefreshCw, Layers, CheckCircle2,
  Car, Bird, Package, Clipboard, Info, ArrowRight, Zap, X, Maximize2
} from "lucide-react";

// --- Types ---
interface GradioImageData { url?: string; path?: string; }
interface GradioImage { 
  image?: GradioImageData; 
  url?: string; 
  path?: string; 
  caption?: string; 
}

interface ParsedStats {
  precision: number; mapR: number;
  recall: { r1: number; r5: number; r10: number; r100: number };
  embeddings: string; dim: number;
}

type LoadResourcesResponse = [string, string];
type RefreshExamplesResponse = [GradioImage[]];
type SearchResponse = [GradioImage[], string];

export default function DinoRetrievalApp() {
  // Config
  const [dataset, setDataset] = useState<string>("Cars196");
  const [size, setSize] = useState<string>("b"); 
  const [isFinetuned, setIsFinetuned] = useState<boolean>(false);
  const [topK, setTopK] = useState<number>(10);

  // UI State
  const [status, setStatus] = useState<string>("Initializing...");
  const [stats, setStats] = useState<ParsedStats | null>(null);
  const [examples, setExamples] = useState<GradioImage[]>([]);
  const [results, setResults] = useState<GradioImage[]>([]);
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isExamplesLoading, setIsExamplesLoading] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"example" | "upload">("example");
  const [showInfo, setShowInfo] = useState<boolean>(true);
  const [selectedImage, setSelectedImage] = useState<GradioImage | null>(null);

  const clientRef = useRef<Client | null>(null);

  // --- Utility Functions ---
  const getImageUrl = (item: GradioImage | null): string => {
    if (!item) return "";
    return item.image?.url || item.image?.path || item.url || item.path || "";
  };

  const parseCaption = (caption?: string) => {
    if (!caption) return { className: "Unknown", sim: "0.000" };
    const parts = caption.split("\n");
    const className = parts[0]?.replace("Class: ", "").replace("Label: ", "") || "Unknown";
    const sim = parts[1]?.replace("Sim: ", "") || "0.000";
    return { className, sim };
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

  // --- API Handlers ---

  // Standalone function to refresh examples
  const refreshExamples = async () => {
    setIsExamplesLoading(true);
    try {
      const client = await getGradioClient();
      const exResult = await client.predict("/refresh_examples_wrapper", {});
      const exData = exResult.data as RefreshExamplesResponse;
      setExamples(exData[0] || []);
    } catch (err) {
      console.error("Failed to refresh examples:", err);
    } finally {
      setIsExamplesLoading(false);
    }
  };

  const loadResources = async () => {
    setIsLoading(true);
    setStatus("Downloading Weights & Mapping...");
    try {
      const client = await getGradioClient();
      const result = await client.predict("/load_resources", {
        dataset, dino_version: "3", dino_size: size, is_finetuned: isFinetuned,
      });
      const data = result.data as LoadResourcesResponse;
      setStatus(data[0]);
      setStats(parseStats(data[1]));
      
      // Fetch initial examples
      await refreshExamples();
    } catch { 
      setStatus("Sync Failed"); 
    } finally { 
      setIsLoading(false); 
    }
  };

  const handleSearch = useCallback(async (source: GradioImage | File | Blob) => {
    setIsSearching(true);
    setStatus("Analyzing Visual Semantics...");
    
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

  useEffect(() => { loadResources(); }, [dataset, size, isFinetuned]);

  const datasets = [
    { id: "Cars196", name: "Cars-196", icon: <Car size={20} />, desc: "196 Car classes" },
    { id: "CUB", name: "CUB-200", icon: <Bird size={20} />, desc: "200 Bird species" },
    { id: "StanfordOnlineProducts", name: "Products", icon: <Package size={20} />, desc: "Online products" },
  ];

  return (
    <div className="min-h-screen bg-[#f1f5f9] text-slate-900 font-sans pb-20 selection:bg-blue-100">
      
      {/* --- INITIALIZE AI FULLSCREEN OVERLAY --- */}
      <AnimatePresence>
        {isLoading && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-slate-950/90 backdrop-blur-xl flex flex-col items-center justify-center text-white p-6"
          >
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
              className="bg-white/5 p-16 rounded-[60px] border border-white/10 flex flex-col items-center gap-10 shadow-2xl max-w-2xl w-full text-center"
            >
              <div className="relative">
                <Loader2 className="animate-spin text-blue-500" size={100} strokeWidth={1} />
                <Zap className="absolute inset-0 m-auto text-white animate-pulse" size={32} />
              </div>
              <div className="space-y-4">
                <h2 className="text-4xl font-black tracking-tighter uppercase italic text-white/90">Loading Resources</h2>
                <p className="text-blue-400 font-mono text-2xl animate-pulse tracking-tight px-4">{status}</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* --- IMAGE EXPAND / LIGHTBOX MODAL --- */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[110] bg-black/95 backdrop-blur-md flex flex-col items-center justify-center p-4 md:p-10"
            onClick={() => setSelectedImage(null)}
          >
            <button className="absolute top-8 right-8 text-white/40 hover:text-white transition-colors">
              <X size={44} />
            </button>
            <motion.div 
              initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }}
              className="relative max-w-4xl w-full flex flex-col items-center"
              onClick={(e) => e.stopPropagation()}
            >
              <img 
                src={getImageUrl(selectedImage)} 
                className="max-h-[70vh] w-auto rounded-[40px] shadow-2xl border border-white/10"
                alt="Expanded View"
              />
              <div className="mt-10 text-center space-y-3">
                <p className="text-blue-500 font-black uppercase tracking-[0.3em] text-xs">Ground Truth Category</p>
                <h2 className="text-white text-4xl font-black tracking-tight max-w-2xl">
                  {parseCaption(selectedImage.caption).className}
                </h2>
                <div className="inline-block px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-white/60 font-mono text-sm">
                  Confidence Metric: {parseCaption(selectedImage.caption).sim}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-7xl mx-auto p-4 md:p-8 space-y-8">
        <header className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
          <div className="space-y-4">
            <motion.h1 initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-5xl font-black tracking-tighter text-slate-900">
              🦖 DINO<span className="text-blue-600">v3</span> Retrieval
            </motion.h1>
            
            <AnimatePresence>
              {showInfo && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                  className="bg-white p-6 rounded-[32px] border border-slate-200 shadow-sm space-y-5 relative overflow-hidden"
                >
                  <button onClick={() => setShowInfo(false)} className="absolute top-5 right-5 p-1 rounded-full hover:bg-slate-100 transition-colors text-slate-300 hover:text-slate-600">
                    <X size={18} />
                  </button>
                  <div className="flex gap-5 pr-8">
                    <div className="bg-blue-50 p-2.5 rounded-2xl h-fit shadow-inner"><Info className="text-blue-500" size={20} /></div>
                    <div className="text-sm text-slate-600 leading-relaxed">
                      <p>
                        Powered by <a href="https://ai.meta.com/dinov3/" target="_href"> <strong className="font-bold text-slate-900">Meta AI&apos;s DINOv3</strong>.</a> In this demo you can select from different image retrieval datasets to explore how well DINOv3 performs in finding visually similar images.
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="flex flex-col md:items-end gap-3">
            <div className="flex items-center gap-4 bg-white p-3.5 rounded-2xl border shadow-sm px-8">
              <div className={`w-3 h-3 rounded-full ${isLoading || isSearching ? 'bg-amber-400 animate-pulse' : 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'}`} />
              <span className="text-xs font-mono font-black text-slate-800 uppercase tracking-tighter">{status}</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar Section */}
          <div className="lg:col-span-4 space-y-6">
            <section className="bg-white p-7 rounded-[40px] shadow-sm border border-slate-200 space-y-8">
              <div className="space-y-3">
                <h3 className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                  <Database size={14} /> Domain Target
                </h3>
                <div className="grid grid-cols-1 gap-2.5 pt-1">
                  {datasets.map((d) => (
                    <button key={d.id} onClick={() => setDataset(d.id)} className={`flex items-center gap-4 p-4 rounded-3xl border-2 transition-all text-left group ${dataset === d.id ? 'border-blue-600 bg-blue-50 shadow-md' : 'border-slate-50 hover:border-slate-200 bg-slate-50/50'}`}>
                      <div className={`p-3 rounded-2xl transition-all group-hover:scale-110 ${dataset === d.id ? 'bg-blue-600 text-white shadow-lg' : 'bg-white text-slate-400 border border-slate-100'}`}>{d.icon}</div>
                      <div>
                        <p className="font-black text-sm text-slate-900 tracking-tight">{d.name}</p>
                        <p className="text-[10px] text-slate-400 font-bold uppercase tracking-tighter">{d.desc}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest text-center block">Model Size</label>
                  <div className="flex p-1 bg-slate-100 rounded-2xl">
                    {['s', 'b'].map(v => (
                      <button key={v} onClick={() => setSize(v)} className={`flex-1 py-2 text-xs font-black rounded-xl transition-all ${size === v ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}>
                        {v === 's' ? 'Small' : 'Base'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest text-center block">Weights</label>
                  <button onClick={() => setIsFinetuned(!isFinetuned)} className={`w-full py-2.5 rounded-2xl border text-[10px] font-black uppercase tracking-tighter transition-all ${isFinetuned ? 'bg-blue-600 text-white border-blue-600 shadow-lg' : 'bg-slate-100 text-slate-400 border-slate-200'}`}>
                    {isFinetuned ? 'Finetuned' : 'Pretrained'}
                  </button>
                </div>
              </div>

              <div className="space-y-4 pt-2 border-t border-slate-100">
                <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-widest">
                  <span>Retrieve Top-K</span> <span className="text-blue-600 font-mono text-xs">{topK}</span>
                </div>
                <input type="range" min="1" max="50" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-blue-600" />
              </div>
            </section>

            {/* Performance Stats Bento Card */}
            <AnimatePresence>
              {stats && (
                <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-blue-800 p-8 rounded-[40px] shadow-2xl text-white space-y-8">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 p-5 rounded-3xl border border-white/10 shadow-inner">
                      <p className="text-[9px] font-black text-white/30 uppercase tracking-[0.2em] mb-1">Precision@1</p>
                      <p className="text-3xl font-black text-blue-400">{stats.precision}%</p>
                    </div>
                    <div className="bg-white/5 p-5 rounded-3xl border border-white/10 shadow-inner">
                      <p className="text-[9px] font-black text-white/30 uppercase tracking-[0.2em] mb-1">Recall@1</p>
                      <p className="text-3xl font-black text-blue-400">{stats.recall.r1}%</p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between text-[10px] font-black text-white/20 uppercase tracking-[0.3em]">
                      <span className="flex items-center gap-2 font-mono"><Target size={12}/> Global MAP@R</span> <span>{stats.mapR}%</span>
                    </div>
                    <div className="h-2.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/5 p-[1px]">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${stats.mapR}%` }} className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.6)]" />
                    </div>
                  </div>
                </motion.section>
              )}
            </AnimatePresence>
          </div>

          {/* Main Interface Section */}
          <div className="lg:col-span-8 space-y-8">
            
            {/* Active Query Feedback */}
            <AnimatePresence>
              {activeQuery && (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-white p-5 rounded-[40px] border border-blue-100 shadow-xl flex items-center gap-8 group">
                  <div className="relative shrink-0">
                    <img src={activeQuery} className="w-24 h-24 rounded-[32px] object-cover shadow-2xl bg-slate-50 border-2 border-slate-100 group-hover:rotate-1 transition-transform" />
                    {isSearching && (
                      <div className="absolute inset-0 bg-blue-600/40 backdrop-blur-[2px] animate-pulse flex items-center justify-center rounded-[32px]">
                        <Loader2 className="animate-spin text-white" size={28} />
                      </div>
                    )}
                  </div>
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center gap-2 text-blue-600 font-black text-[10px] uppercase tracking-widest">
                      <Zap size={14} className="fill-blue-600" /> Neural Match in Progress
                    </div>
                    <p className="text-slate-900 font-black text-2xl tracking-tight leading-none ">Computing Vector Similarity</p>
                  </div>
                  <div className="mr-6 text-slate-100 hidden md:block">
                    <ArrowRight size={56} strokeWidth={3} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="bg-white rounded-[40px] shadow-sm border border-slate-200 overflow-hidden">
              <div className="flex p-2 bg-slate-50/50 m-2 rounded-[28px] border border-slate-100">
                {(["example", "upload"] as const).map((tab) => (
                  <button key={tab} onClick={() => setActiveTab(tab)} className={`flex-1 py-3 text-xs font-black uppercase tracking-widest rounded-2xl transition-all ${activeTab === tab ? 'bg-white shadow-xl text-blue-600 border border-slate-100' : 'text-slate-400 hover:text-slate-600'}`}>
                    {tab === 'example' ? 'Examples' : 'Custom Image'}
                  </button>
                ))}
              </div>

              <div className="p-8 pt-4">
                {activeTab === "example" ? (
                  <div className="relative min-h-[160px] space-y-6">
                    {/* Header for Examples with Refresh Button */}
                    <div className="flex items-center justify-between px-2">
                      <h4 className="text-[10px] font-black uppercase text-slate-400 tracking-widest flex items-center gap-2">
                        Test Gallery
                      </h4>
                      <button 
                        onClick={refreshExamples}
                        disabled={isExamplesLoading}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 text-[10px] font-black uppercase tracking-widest transition-all active:scale-95 disabled:opacity-50"
                      >
                        <RefreshCw size={14} className={isExamplesLoading ? "animate-spin" : ""} />
                        Get New Examples
                      </button>
                    </div>

                    <AnimatePresence mode="wait">
                      {isExamplesLoading ? (
                        <motion.div key="loader" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center justify-center gap-4 py-12">
                           <Loader2 className="animate-spin text-slate-300" size={40} />
                           <span className="text-[10px] font-black uppercase text-slate-300 tracking-[0.3em]">Randomizing Samples</span>
                        </motion.div>
                      ) : (
                        <motion.div key="grid" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                          {examples.map((item, idx) => (
                            <motion.img 
                              key={idx} whileHover={{ scale: 1.08, zIndex: 10 }} whileTap={{ scale: 0.95 }} 
                              src={getImageUrl(item)} 
                              className="w-full aspect-square object-cover rounded-3xl cursor-pointer shadow-sm border-2 border-transparent hover:border-blue-400 transition-all bg-slate-100" 
                              onClick={() => handleSearch(item)} 
                            />
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ) : (
                  <div className="border-2 border-dashed border-slate-200 rounded-[40px] p-12 text-center space-y-5 hover:border-blue-400 transition-all bg-slate-50/50 group">
                    <div className="flex flex-wrap justify-center gap-6">
                      <div className="bg-white p-6 rounded-[32px] shadow-sm border border-slate-100 flex flex-col items-center gap-3 transition-all group-hover:shadow-lg group-hover:-translate-y-1">
                        <div className="bg-blue-50 p-2.5 rounded-2xl"><Clipboard className="text-blue-500" size={24} /></div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Ctrl+V Paste</span>
                      </div>
                      <label className="bg-white p-6 rounded-[32px] shadow-sm border border-slate-100 flex flex-col items-center gap-3 cursor-pointer transition-all group-hover:shadow-lg group-hover:-translate-y-1">
                        <div className="bg-blue-50 p-2.5 rounded-2xl"><Search className="text-blue-500" size={24} /></div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Local File</span>
                        <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && handleSearch(e.target.files[0])} className="hidden" />
                      </label>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Results Section */}
            <section className="space-y-6">
              <h3 className="text-2xl font-black text-slate-900 flex items-center gap-3 px-2">Matches Found <span className="text-sm font-bold bg-slate-200 px-3 py-1 rounded-full text-slate-500">{results.length}</span></h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
                {results.length > 0 ? results.map((res, idx) => {
                  const { className, sim } = parseCaption(res.caption);
                  return (
                    <motion.div 
                      initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.04 }} 
                      key={idx} 
                      className="bg-white p-2.5 rounded-[32px] shadow-sm border border-slate-100 group cursor-pointer relative"
                      onClick={() => setSelectedImage(res)}
                    >
                      <div className="relative overflow-hidden rounded-[24px] aspect-square bg-slate-50">
                        <img 
                          src={getImageUrl(res)} 
                          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700 shadow-inner" 
                        />
                        <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                          <Maximize2 className="text-white drop-shadow-lg" size={28} />
                        </div>
                        <div className="absolute bottom-2 right-2 bg-blue-600 text-white font-mono text-[9px] px-2 py-0.5 rounded-full shadow-lg">
                          {sim}
                        </div>
                      </div>
                      <div className="mt-3 px-2 pb-1">
                        <p className="text-[11px] font-black text-slate-800 leading-[1.1] line-clamp-2 min-h-[2.2em]">
                          {className}
                        </p>
                      </div>
                    </motion.div>
                  );
                }) : (
                  <div className="col-span-full py-32 text-center text-slate-300 flex flex-col items-center gap-6 bg-white rounded-[40px] border border-dashed border-slate-200">
                    <div className="bg-slate-50 p-6 rounded-full ring-8 ring-slate-50/50"><Search size={48} className="text-slate-100" /></div>
                    <p className="font-black text-slate-400 uppercase tracking-[0.3em] text-xs">Waiting for Query</p>
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