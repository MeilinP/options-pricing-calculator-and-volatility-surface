import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Calculator, TrendingUp, Activity } from 'lucide-react';

// Black-Scholes formula implementation
const normalCDF = (x) => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - prob : prob;
};

const blackScholes = (S, K, T, r, sigma, type = 'call') => {
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  
  if (type === 'call') {
    return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
  } else {
    return K * Math.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
  }
};

// Greeks calculation
const calculateGreeks = (S, K, T, r, sigma, type = 'call') => {
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const pdf = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  
  const delta = type === 'call' ? normalCDF(d1) : normalCDF(d1) - 1;
  const gamma = pdf(d1) / (S * sigma * Math.sqrt(T));
  const vega = S * pdf(d1) * Math.sqrt(T) / 100;
  const theta = type === 'call' 
    ? (-S * pdf(d1) * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normalCDF(d2)) / 365
    : (-S * pdf(d1) * sigma / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normalCDF(-d2)) / 365;
  const rho = type === 'call'
    ? K * T * Math.exp(-r * T) * normalCDF(d2) / 100
    : -K * T * Math.exp(-r * T) * normalCDF(-d2) / 100;
  
  return { delta, gamma, vega, theta, rho };
};

// Monte Carlo simulation
const monteCarloSimulation = (S, K, T, r, sigma, type = 'call', simulations = 10000, steps = 252) => {
  const dt = T / steps;
  let payoffSum = 0;

  for (let i = 0; i < simulations; i++) {
    let St = S;
    const pathPrices = [St];

    for (let j = 0; j < steps; j++) {
      const Z = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
      St *= Math.exp((r - 0.5 * sigma ** 2) * dt + sigma * Math.sqrt(dt) * Z);
      pathPrices.push(St);
    }

    const ST = pathPrices[pathPrices.length - 1];
    let payoff = 0;

    if (type === 'call')         payoff = Math.max(ST - K, 0);
    else if (type === 'put')     payoff = Math.max(K - ST, 0);
    else if (type === 'asian_call') payoff = Math.max((pathPrices.reduce((a,b)=>a+b,0)/pathPrices.length) - K, 0);
    else if (type === 'asian_put')  payoff = Math.max(K - (pathPrices.reduce((a,b)=>a+b,0)/pathPrices.length), 0);

    payoffSum += payoff;
  }

  return Math.exp(-r * T) * (payoffSum / simulations);
};

// Binomial tree pricing
const binomialTree = (S, K, T, r, sigma, type = 'call', steps = 100) => {
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const p = (Math.exp(r * dt) - d) / (u - d);
  
  const prices = new Array(steps + 1);
  const values = new Array(steps + 1);
  
  for (let i = 0; i <= steps; i++) {
    prices[i] = S * Math.pow(u, steps - i) * Math.pow(d, i);
    values[i] = type === 'call' 
      ? Math.max(prices[i] - K, 0)
      : Math.max(K - prices[i], 0);
  }
  
  for (let j = steps - 1; j >= 0; j--) {
    for (let i = 0; i <= j; i++) {
      values[i] = Math.exp(-r * dt) * (p * values[i] + (1 - p) * values[i + 1]);
      prices[i] = S * Math.pow(u, j - i) * Math.pow(d, i);
      const exerciseValue = type === 'call'
        ? Math.max(prices[i] - K, 0)
        : Math.max(K - prices[i], 0);
      values[i] = Math.max(values[i], exerciseValue);
    }
  }
  
  return values[0];
};

function App() {
  const [params, setParams] = useState({
    S: 100,
    K: 100,
    T: 1,
    r: 0.05,
    sigma: 0.2,
    type: 'call'
  });

  const [activeTab, setActiveTab] = useState('calculator');

  const results = useMemo(() => {
    const bs = blackScholes(params.S, params.K, params.T, params.r, params.sigma, params.type);
    const greeks = calculateGreeks(params.S, params.K, params.T, params.r, params.sigma, params.type);
    const mc = monteCarloSimulation(params.S, params.K, params.T, params.r, params.sigma, params.type);
    const bt = binomialTree(params.S, params.K, params.T, params.r, params.sigma, params.type);
    
    return { bs, greeks, mc, bt };
  }, [params]);

  const sensitivityData = useMemo(() => {
    const data = [];
    const range = Array.from({ length: 41 }, (_, i) => params.S * (0.7 + i * 0.015));
    
    range.forEach(spot => {
      data.push({
        spot: spot.toFixed(2),
        price: blackScholes(spot, params.K, params.T, params.r, params.sigma, params.type),
        delta: calculateGreeks(spot, params.K, params.T, params.r, params.sigma, params.type).delta
      });
    });
    
    return data;
  }, [params]);

  const volatilitySurfaceData = useMemo(() => {
    const data = [];
    const volRange = Array.from({ length: 21 }, (_, i) => 0.1 + i * 0.02);
    
    volRange.forEach(vol => {
      data.push({
        volatility: (vol * 100).toFixed(0) + '%',
        price: blackScholes(params.S, params.K, params.T, params.r, vol, params.type)
      });
    });
    
    return data;
  }, [params]);

  const updateParam = (key, value) => {
    setParams(prev => ({ ...prev, [key]: parseFloat(value) || value }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <Calculator className="w-10 h-10 text-blue-400" />
            Options Pricing Calculator
          </h1>
          <p className="text-slate-300">Black-Scholes | Monte Carlo Simulation | Binomial Tree | Greeks Analysis</p>
        </header>

        {/* Navigation tabs */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {[
            { id: 'calculator', label: 'Calculator', icon: Calculator },
            { id: 'sensitivity', label: 'Sensitivity Analysis', icon: TrendingUp },
            { id: 'volatility', label: 'Volatility Analysis', icon: Activity }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Parameter input panel */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
              <h2 className="text-xl font-bold mb-4 text-blue-400">Input Parameters</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Option Type</label>
                  <select
                    value={params.type}
                    onChange={(e) => updateParam('type', e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="call">Call Option</option>
                    <option value="put">Put Option</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Spot Price (S): ${params.S}
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    step="1"
                    value={params.S}
                    onChange={(e) => updateParam('S', e.target.value)}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Strike Price (K): ${params.K}
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    step="1"
                    value={params.K}
                    onChange={(e) => updateParam('K', e.target.value)}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Time to Maturity (T): {params.T.toFixed(2)} years
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={params.T}
                    onChange={(e) => updateParam('T', e.target.value)}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Risk-Free Rate (r): {(params.r * 100).toFixed(1)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="0.15"
                    step="0.005"
                    value={params.r}
                    onChange={(e) => updateParam('r', e.target.value)}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Volatility (σ): {(params.sigma * 100).toFixed(1)}%
                  </label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.8"
                    step="0.01"
                    value={params.sigma}
                    onChange={(e) => updateParam('sigma', e.target.value)}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Results display area */}
          <div className="lg:col-span-2">
            {activeTab === 'calculator' && (
              <div className="space-y-6">
                {/* Pricing results */}
                <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
                  <h2 className="text-xl font-bold mb-4 text-blue-400">Pricing Results</h2>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-slate-700 rounded-lg p-4">
                      <div className="text-sm text-slate-400 mb-1">Black-Scholes</div>
                      <div className="text-2xl font-bold text-green-400">
                        ${results.bs.toFixed(4)}
                      </div>
                    </div>
                    <div className="bg-slate-700 rounded-lg p-4">
                      <div className="text-sm text-slate-400 mb-1">Monte Carlo</div>
                      <div className="text-2xl font-bold text-blue-400">
                        ${results.mc.toFixed(4)}
                      </div>
                    </div>
                    <div className="bg-slate-700 rounded-lg p-4">
                      <div className="text-sm text-slate-400 mb-1">Binomial Tree</div>
                      <div className="text-2xl font-bold text-purple-400">
                        ${results.bt.toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Greeks */}
                <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
                  <h2 className="text-xl font-bold mb-4 text-blue-400">Greeks</h2>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {Object.entries(results.greeks).map(([key, value]) => (
                      <div key={key} className="bg-slate-700 rounded-lg p-4">
                        <div className="text-xs text-slate-400 uppercase mb-1">{key}</div>
                        <div className="text-lg font-bold">{value.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-4 text-sm text-slate-400 space-y-1">
                    <p><strong>Delta:</strong> Rate of change of option price with respect to underlying price</p>
                    <p><strong>Gamma:</strong> Rate of change of delta</p>
                    <p><strong>Vega:</strong> Sensitivity to volatility (per 1% change)</p>
                    <p><strong>Theta:</strong> Time decay (per day)</p>
                    <p><strong>Rho:</strong> Sensitivity to interest rate (per 1% change)</p>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'sensitivity' && (
              <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
                <h2 className="text-xl font-bold mb-4 text-blue-400">Spot Price Sensitivity Analysis</h2>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={sensitivityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="spot" stroke="#94a3b8" label={{ value: 'Spot Price ($)', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} name="Option Price" />
                    <Line type="monotone" dataKey="delta" stroke="#10b981" strokeWidth={2} name="Delta" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {activeTab === 'volatility' && (
              <div className="bg-slate-800 rounded-xl p-6 shadow-xl">
                <h2 className="text-xl font-bold mb-4 text-blue-400">Volatility Sensitivity Analysis</h2>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={volatilitySurfaceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="volatility" stroke="#94a3b8" label={{ value: 'Volatility', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'Option Price ($)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="price" stroke="#8b5cf6" strokeWidth={2} name="Option Price" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        {/* Project description */}
        <div className="mt-8 bg-slate-800 rounded-xl p-6 shadow-xl">
          <h3 className="text-lg font-bold mb-3 text-blue-400">About This Project</h3>
          <div className="text-sm text-slate-300 space-y-2">
            <p>• <strong>Black-Scholes Model:</strong> Classic analytical formula for European options pricing</p>
            <p>• <strong>Monte Carlo Simulation:</strong> Stochastic approach using random price path simulations</p>
            <p>• <strong>Binomial Tree:</strong> Discrete-time model suitable for American options with early exercise</p>
            <p>• <strong>Greeks:</strong> Risk measures showing sensitivity to various market parameters</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
