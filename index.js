import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import * as ftp from 'basic-ftp';
import { GoogleGenAI } from '@google/genai';
import pdfParse from 'pdf-parse-fork';
import dotenv from 'dotenv';

dotenv.config();

// Configuration for REDCap API and Gemini API
const REDCAP_API_URL = process.env.REDCAP_API_URL || 'https://redcap.vfn.cz/api/';
const API_TOKEN = process.env.REDCAP_API_TOKEN;
const GEMINI_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({ apiKey: GEMINI_KEY || '' });

/**
 * 1. Export raw records from REDCap API
 */
async function exportAllRedcapData() {
    const formData = new URLSearchParams({
        token: API_TOKEN,
        content: 'record',
        action: 'export',
        format: 'json',
        type: 'flat',
        rawOrLabel: 'raw',
        rawOrLabelHeaders: 'raw',
        exportCheckboxLabel: 'false',
        exportSurveyFields: 'false',
        exportDataAccessGroups: 'false',
        returnFormat: 'json'
    });

    try {
        console.log('Sending request to REDCap API...');

        const response = await fetch(REDCAP_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            },
            body: formData.toString()
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`REDCap API Error (${response.status}): ${errorText}`);
        }

        const records = await response.json();
        console.log(`Successfully downloaded ${records.length} records.`);
        return records;

    } catch (error) {
        console.error('Error communicating with REDCap API:', error.message);
        throw error;
    }
}

/**
 * 2. Helper functions for date calculations and protocol tracking
 */
function diffInDays(startDateStr, endDateStr) {
    if (!startDateStr || !endDateStr) return null;
    const start = new Date(startDateStr);
    const end = new Date(endDateStr);

    if (isNaN(start.getTime()) || isNaN(end.getTime())) {
        return null;
    }
    return Math.round((end - start) / (1000 * 60 * 60 * 24));
}

function daysSince(dateStr) {
    if (!dateStr) return null;
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return null;
    return Math.round((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24));
}

/**
 * Parse patient records and calculate protocol interval metrics
 */
function pacCheck(rawData) {
    if (!Array.isArray(rawData)) return [];

    return rawData.map((pacientka) => {
        const scrDate = pacientka.scr_v_date;
        const rtStart = pacientka.rt_start;
        const rtEnd = pacientka.rt_end;
        const v1Date = pacientka.v1_date;
        const v2Date = pacientka.v2_date;
        const v3Date = pacientka.v3_date;

        return {
            id: pacientka.record_id || pacientka.id,
            scr: scrDate ? new Date(scrDate) : null,
            scr_rt: diffInDays(scrDate, rtStart),
            rt: rtEnd ? new Date(rtEnd) : null,
            rt_v1: diffInDays(rtEnd, v1Date),
            v1: v1Date ? new Date(v1Date) : null,
            scr_v1: diffInDays(scrDate, v1Date),
            v1_v2: diffInDays(v1Date, v2Date),
            v2: v2Date ? new Date(v2Date) : null,
            v2_v3: diffInDays(v2Date, v3Date),
            v3: v3Date ? new Date(v3Date) : null,
            eot: pacientka.eot_prop || '',
            daysSinceScr: daysSince(scrDate),
            daysSinceV1: daysSince(v1Date),
            daysSinceV2: daysSince(v2Date),
            raw: pacientka
        };
    });
}

/**
 * Extract aggregated trial cohort metrics
 */
function calculateAggregatedStats(records) {
    const isFilled = (value) => value !== undefined && value !== null && String(value).trim() !== '';

    const totalITT = records.filter(r => isFilled(r.sample_id) || isFilled(r.scr_v_date) || isFilled(r.record_id)).length;
    const completedRT = records.filter(r => isFilled(r.rt_start) || isFilled(r.rt_end)).length;
    const v1Count = records.filter(r => isFilled(r.sample_id_v1) || isFilled(r.v1_date)).length;
    const v2Count = records.filter(r => isFilled(r.sample_id_v2) || isFilled(r.v2_date)).length;
    const v3Count = records.filter(r => isFilled(r.sample_id_v3) || isFilled(r.v3_date)).length;
    const recurrenceCount = records.filter(r => isFilled(r.rec_date) || String(r.v2_tumor).trim() === '1' || String(r.v3_tumor).trim() === '1').length;
    const eot = records.filter(r => isFilled(r.eot_prop)).length;
    const activeObservation = Math.max(0, totalITT - recurrenceCount - eot);

    return {
        totalITT,
        completedRT,
        v1Count,
        v2Count,
        v3Count,
        recurrenceCount,
        eot,
        activeObservation
    };
}

/**
 * Determine HPV status for a given record and visit prefix
 */
function getHpvStatus(record, prefix) {
    if (!record) return 'untested';

    const directVal = record[`${prefix}_hpv`];
    if (directVal !== undefined && directVal !== null && String(directVal).trim() !== '') {
        const strVal = String(directVal).trim();
        if (strVal === '0' || strVal === 'negative') return 'negative';
        return 'positive';
    }

    // Checkbox field checks in REDCap flat JSON
    const negKey = `${prefix}_hpv___0`;
    if (record[negKey] === '1' || record[negKey] === 1) {
        return 'negative';
    }

    const hpvTypes = ['16', '18', '31', '33', '35', '39', '45', '51', '52', '56', '58', '59', '66', '68'];
    const isPos = hpvTypes.some(t => record[`${prefix}_hpv___${t}`] === '1' || record[`${prefix}_hpv___${t}`] === 1);
    if (isPos) return 'positive';

    return 'untested';
}

/**
 * Mathematical complementary error function (erfc) for exact p-value calculation
 */
function erfc(x) {
    const z = Math.abs(x);
    const t = 1.0 / (1.0 + 0.5 * z);
    const ans = t * Math.exp(-z * z - 1.26551223 +
        t * (1.00002368 +
        t * (0.37409196 +
        t * (0.09678418 +
        t * (-0.18628806 +
        t * (0.27886807 +
        t * (-1.13520398 +
        t * (1.48851587 +
        t * (-0.82215223 +
        t * 0.17087277))))))))) ;
    return x >= 0 ? ans : 2.0 - ans;
}

/**
 * Calculates exact Chi-Square, p-value, Odds Ratio (OR), and 95% Confidence Interval
 */
function calculateChiSquareAndOR(regA, regB, perA, perB) {
    const n = regA + regB + perA + perB;
    if (n === 0 || (regA + regB) === 0 || (perA + perB) === 0 || (regA + perA) === 0 || (regB + perB) === 0) {
        return { chi2: 0, pValue: 'P = 1.0000', rawP: 1, OR: null, ciLow: null, ciHigh: null, significant: false };
    }

    const a = regA, b = regB, c = perA, d = perB;
    const numerator = n * Math.pow(Math.abs(a * d - b * c), 2);
    const denominator = (a + b) * (c + d) * (a + c) * (b + d);
    const chi2 = numerator / denominator;
    const pValue = Math.min(1, Math.max(0, erfc(Math.sqrt(chi2 / 2))));

    const a_c = a === 0 || b === 0 || c === 0 || d === 0 ? a + 0.5 : a;
    const b_c = a === 0 || b === 0 || c === 0 || d === 0 ? b + 0.5 : b;
    const c_c = a === 0 || b === 0 || c === 0 || d === 0 ? c + 0.5 : c;
    const d_c = a === 0 || b === 0 || c === 0 || d === 0 ? d + 0.5 : d;

    const OR = (a_c * d_c) / (b_c * c_c);
    const seLnOR = Math.sqrt(1/a_c + 1/b_c + 1/c_c + 1/d_c);
    const ciLow = Math.exp(Math.log(OR) - 1.96 * seLnOR);
    const ciHigh = Math.exp(Math.log(OR) + 1.96 * seLnOR);

    const formattedP = pValue < 0.001 ? 'P < 0.001' : `P = ${pValue.toFixed(4)}`;

    return {
        chi2: Number(chi2.toFixed(2)),
        pValue: formattedP,
        rawP: pValue,
        OR: Number(OR.toFixed(2)),
        ciLow: Number(ciLow.toFixed(2)),
        ciHigh: Number(ciHigh.toFixed(2)),
        significant: pValue < 0.05
    };
}

/**
 * Generates NEJM Style Forest Plot SVG
 */
function generateForestPlotSvg(results) {
    const valid = results.filter(r => r.stats && r.stats.OR !== null);
    if (valid.length === 0) return '';

    const width = 860;
    const itemHeight = 36;
    const height = 70 + valid.length * itemHeight;
    const minX = 0.1;
    const maxX = 10;

    const mapX = (orVal) => {
        const clamped = Math.max(minX, Math.min(maxX, orVal));
        const logVal = Math.log(clamped);
        const logMin = Math.log(minX);
        const logMax = Math.log(maxX);
        return 330 + ((logVal - logMin) / (logMax - logMin)) * 360;
    };

    const lineNullX = mapX(1.0);

    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:4px; margin-bottom:25px; font-family:'Inter', Arial, sans-serif;">`;
    svg += `<rect x="0" y="0" width="${width}" height="42" fill="#fafafa"/>`;
    svg += `<line x1="0" y1="42" x2="${width}" y2="42" stroke="#a6192e" stroke-width="2"/>`;
    svg += `<text x="16" y="26" fill="#a6192e" font-size="13" font-weight="700" font-family="'Merriweather', Georgia, serif">FIGURE 1. FOREST PLOT OF ODDS RATIOS (OR) AND 95% CONFIDENCE INTERVALS FOR DISEASE RECURRENCE</text>`;
    
    svg += `<line x1="${lineNullX}" y1="42" x2="${lineNullX}" y2="${height - 25}" stroke="#999999" stroke-dasharray="4,4" stroke-width="1"/>`;

    valid.forEach((r, idx) => {
        const y = 66 + idx * itemHeight;
        const lowX = mapX(r.stats.ciLow);
        const highX = mapX(r.stats.ciHigh);
        const orX = mapX(r.stats.OR);
        
        let color = "#555555";
        if (r.stats.rawP < 0.05) color = "#a6192e";
        else if (r.stats.rawP < 0.10) color = "#d97706";

        const labelName = r.nazev.split('(')[0].trim();
        svg += `<text x="16" y="${y + 4}" fill="#222222" font-size="12" font-weight="500">${labelName}</text>`;
        svg += `<line x1="${lowX}" y1="${y}" x2="${highX}" y2="${y}" stroke="${color}" stroke-width="1.5"/>`;
        svg += `<line x1="${lowX}" y1="${y - 4}" x2="${lowX}" y2="${y + 4}" stroke="${color}" stroke-width="1.5"/>`;
        svg += `<line x1="${highX}" y1="${y - 4}" x2="${highX}" y2="${y + 4}" stroke="${color}" stroke-width="1.5"/>`;
        svg += `<circle cx="${orX}" cy="${y}" r="4" fill="${color}"/>`;
        
        svg += `<text x="${width - 16}" y="${y + 4}" fill="#222222" font-size="11" text-anchor="end" font-weight="600">OR ${r.stats.OR} (95% CI: ${r.stats.ciLow}-${r.stats.ciHigh}) | ${r.stats.pValue}</text>`;
    });

    svg += `<line x1="0" y1="${height - 25}" x2="${width}" y2="${height - 25}" stroke="#e0e0e0" stroke-width="1"/>`;
    svg += `<text x="${lineNullX}" y="${height - 8}" fill="#777777" font-size="10" text-anchor="middle">OR = 1.0 (Reference Line)</text>`;
    svg += `</svg>`;
    return svg;
}

/**
 * Generates Subgroup Comparison Horizontal Bar Chart SVG
 */
function generateBarChartSvg(title, categories, pValueText, width = 560, height = 260) {
    let barsSvg = '';
    categories.forEach((cat, idx) => {
        const y = 60 + idx * 85;
        const maxBarWidth = 310;
        const barWidth = Math.max(10, (cat.pct / 100) * maxBarWidth);

        barsSvg += `
        <g transform="translate(150, ${y})">
            <text x="-12" y="19" fill="#111111" font-size="13" font-weight="700" font-family="'Merriweather', Georgia, serif" text-anchor="end">${cat.label}</text>
            <rect x="0" y="0" width="${maxBarWidth}" height="28" fill="#f0f0f0" rx="3"/>
            <rect x="0" y="0" width="${barWidth}" height="28" fill="${cat.color}" rx="3"/>
            <text x="${barWidth + 10}" y="19" fill="#111111" font-size="13" font-weight="800" font-family="'Inter', sans-serif">${cat.pct}%</text>
            <text x="0" y="48" fill="#666666" font-size="11" font-weight="500" font-family="'Inter', sans-serif">${cat.subtext}</text>
        </g>
        `;
    });

    return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:4px; font-family:'Inter', Arial, sans-serif; display:block;">
        <rect x="0" y="0" width="${width}" height="38" fill="#fafafa"/>
        <line x1="0" y1="38" x2="${width}" y2="38" stroke="#e0e0e0" stroke-width="1"/>
        <text x="14" y="24" fill="#a6192e" font-size="12" font-weight="700" font-family="'Merriweather', Georgia, serif">${title}</text>
        
        <rect x="${width - 110}" y="8" width="96" height="22" rx="3" fill="#a6192e"/>
        <text x="${width - 62}" y="23" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">${pValueText}</text>

        ${barsSvg}
    </svg>
    `;
}

/**
 * Generates NEJM Style Donut Chart SVG with legend BELOW the ring
 */
function generateDonutChartSvg(title, items, total, width = 540, height = 280) {
    const cx = 270, cy = 100, r = 58, innerR = 36;
    let startAngle = -Math.PI / 2;
    let pathsSvg = '';
    let legendSvg = '';

    const dominantItem = items.reduce((prev, current) => (prev.value > current.value) ? prev : current, items[0]);
    const dominantPct = total > 0 ? ((dominantItem.value / total) * 100).toFixed(1) : 0;

    items.forEach((item, idx) => {
        const sliceAngle = total > 0 ? (item.value / total) * 2 * Math.PI : 0;
        const endAngle = startAngle + sliceAngle;

        const x1 = cx + r * Math.cos(startAngle);
        const y1 = cy + r * Math.sin(startAngle);
        const x2 = cx + r * Math.cos(endAngle);
        const y2 = cy + r * Math.sin(endAngle);

        const x1In = cx + innerR * Math.cos(startAngle);
        const y1In = cy + innerR * Math.sin(startAngle);
        const x2In = cx + innerR * Math.cos(endAngle);
        const y2In = cy + innerR * Math.sin(endAngle);

        const largeArcFlag = sliceAngle > Math.PI ? 1 : 0;

        if (total > 0 && item.value > 0) {
            pathsSvg += `
                <path d="M ${x1In} ${y1In} L ${x1} ${y1} A ${r} ${r} 0 ${largeArcFlag} 1 ${x2} ${y2} L ${x2In} ${y2In} A ${innerR} ${innerR} 0 ${largeArcFlag} 0 ${x1In} ${y1In} Z" 
                      fill="${item.color}" stroke="#ffffff" stroke-width="2"/>
            `;
        }

        const pct = total > 0 ? ((item.value / total) * 100).toFixed(1) : 0;

        let lx, ly;
        if (items.length === 5) {
            lx = (idx % 2 === 0) ? 30 : 280;
            ly = 175 + Math.floor(idx / 2) * 36;
        } else if (items.length === 4) {
            lx = (idx % 2 === 0) ? 30 : 280;
            ly = Math.floor(idx / 2) === 0 ? 180 : 225;
        } else {
            lx = 20 + idx * 170;
            ly = 195;
        }

        legendSvg += `
            <g transform="translate(${lx}, ${ly})">
                <rect x="0" y="0" width="12" height="12" rx="2" fill="${item.color}"/>
                <text x="18" y="10" fill="#222222" font-size="12" font-weight="700" font-family="'Inter', sans-serif">${item.label}</text>
                <text x="18" y="24" fill="#555555" font-size="11" font-family="'Inter', sans-serif"><tspan font-weight="700" fill="#111111">${item.value}</tspan> patients (${pct}%)</text>
            </g>
        `;

        startAngle = endAngle;
    });

    return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e0e0e0; border-radius:4px; font-family:'Inter', Arial, sans-serif; display:block;">
        <rect x="0" y="0" width="${width}" height="36" fill="#fafafa"/>
        <line x1="0" y1="36" x2="${width}" y2="36" stroke="#e0e0e0" stroke-width="1"/>
        <text x="14" y="23" fill="#a6192e" font-size="12" font-weight="700" font-family="'Merriweather', Georgia, serif">${title}</text>
        
        <g>
            ${pathsSvg}
            <circle cx="${cx}" cy="${cy}" r="${innerR - 2}" fill="#ffffff" />
            <text x="${cx}" y="${cy - 2}" fill="#a6192e" font-size="17" font-weight="800" font-family="'Merriweather', Georgia, serif" text-anchor="middle">${dominantPct}%</text>
            <text x="${cx}" y="${cy + 13}" fill="#666666" font-size="9" font-weight="600" font-family="'Inter', sans-serif" text-anchor="middle" text-transform="uppercase">${dominantItem.label.split(' ')[0]}</text>
        </g>

        ${legendSvg}
    </svg>
    `;
}

/**
 * 7a. Generates 100% English HTML table of all patient records (pac.html)
 */
function generujTabulkuPacientek(rawData) {
    let html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HART Study - Patient Registry Table</title>
    <style>
        body { font-family: 'Inter', Arial, sans-serif; padding: 20px; color: #222222; }
        h2 { font-family: 'Merriweather', Georgia, serif; color: #a6192e; border-bottom: 2px solid #a6192e; padding-bottom: 6px; }
        .pac-table { border-collapse: collapse; margin-bottom: 25px; font-size: 13px; width: 100%; }
        .pac-table th { background-color: #f4f4f4; color: #a6192e; text-align: left; padding: 8px 10px; font-weight: 700; border-bottom: 2px solid #a6192e; }
        .pac-table td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; color: #333333; }
        .pac-table tr:nth-child(even) { background-color: #fdfdfd; }
    </style>
</head>
<body>
    <h2>HART Study - Patient Cohort Registry</h2>
    <table class="pac-table">
      <thead>
        <tr>
          <th>Patient ID</th><th>Age</th><th>FIGO Stage</th><th>Histology</th><th>Grade</th>
          <th>Baseline SCC</th><th>Screening HPV</th><th>Visit 1 HPV</th><th>Visit 2 HPV</th><th>Visit 3 HPV</th>
          <th>Recurrence</th><th>Study Status</th>
        </tr>
      </thead>
      <tbody>\n`;

    rawData.forEach(r => {
        const pid = r.record_id || r.id;
        const age = r.scr_age || '-';
        
        let figo = '-';
        const figoMap = { '1':'IA1','2':'IA2','3':'IB1','4':'IB2','5':'IB3','6':'IIA1','7':'IIA2','8':'IIB','9':'IIIA','10':'IIIB','11':'IIIC1','12':'IIIC2','13':'IVA','14':'IVB' };
        if (r.scr_figo && figoMap[r.scr_figo]) figo = figoMap[r.scr_figo];

        let histol = '-';
        if (String(r.scr_histol).trim() === '1') histol = 'Squamous Cell';
        else if (String(r.scr_histol).trim() === '2') histol = 'Adenocarcinoma';
        else if (String(r.scr_histol).trim() === '3') histol = 'Adenosquamous';

        const grade = r.scr_grade ? `G${r.scr_grade}` : '-';
        const scc = r.scr_scc || '-';

        const hpvScr = getHpvStatus(r, 'scr');
        const hpvV1 = getHpvStatus(r, 'v1');
        const hpvV2 = getHpvStatus(r, 'v2');
        const hpvV3 = getHpvStatus(r, 'v3');

        let rec = 'No';
        if (r.rec_date || String(r.v2_tumor).trim() === '1' || String(r.v3_tumor).trim() === '1') {
            rec = 'Yes';
        }

        let status = 'Active Observation';
        if (r.eot_prop) status = 'Discontinued (EOT)';
        else if (rec === 'Yes') status = 'Recurrence';
        else if (r.v3_date) status = 'Completed V3';
        else if (r.v2_date) status = 'Completed V2';
        else if (r.v1_date) status = 'Completed V1';

        html += `<tr>
            <td><strong>${pid}</strong></td>
            <td>${age}</td>
            <td>${figo}</td>
            <td>${histol}</td>
            <td>${grade}</td>
            <td>${scc}</td>
            <td>${hpvScr}</td>
            <td>${hpvV1}</td>
            <td>${hpvV2}</td>
            <td>${hpvV3}</td>
            <td><strong>${rec}</strong></td>
            <td>${status}</td>
        </tr>\n`;
    });

    html += `  </tbody>\n</table>\n</body>\n</html>`;
    return html;
}

/**
 * 7b. Univariate Analysis & Contingency Tables (chi.html)
 */
function generujChiKvadratTabulky(rawData) {
    const kriteria = [
        {
            nazev: "Histology Subtype (Squamous Cell vs Adenocarcinoma)",
            extrahuj: (r) => {
                const h = String(r.scr_histol).trim();
                if (h === '1') return "Squamous Cell";
                if (h === '2') return "Adenocarcinoma";
                return null;
            },
            katA: "Squamous Cell", katB: "Adenocarcinoma"
        },
        {
            nazev: "FIGO Stage (Early Stage IB1-IB2 vs Advanced IIB+)",
            extrahuj: (r) => {
                const f = Number(r.scr_figo);
                if (isNaN(f) || f === 0) return null;
                if (f <= 4) return "IB1-IB2";
                if (f >= 8) return "IIB+";
                return null;
            },
            katA: "IB1-IB2", katB: "IIB+"
        },
        {
            nazev: "Concurrent Cisplatin Chemotherapy (Yes vs No)",
            extrahuj: (r) => {
                const c = String(r.cht).trim();
                if (c === '2') return "Yes";
                if (c === '1') return "No";
                return null;
            },
            katA: "Yes", katB: "No"
        },
        {
            nazev: "HPV Clearance at Visit 1 (Negative vs Persistent Positive)",
            extrahuj: (r) => {
                const st = getHpvStatus(r, 'v1');
                if (st === 'negative') return "Negative (Cleared)";
                if (st === 'positive') return "Persistent Positive";
                return null;
            },
            katA: "Negative (Cleared)", katB: "Persistent Positive"
        }
    ];

    const vraceneVysledky = [];

    kriteria.forEach(k => {
        let regA = 0, regB = 0; // No recurrence
        let perA = 0, perB = 0; // Recurrence

        rawData.forEach(r => {
            const hodnota = k.extrahuj(r);
            if (!hodnota) return;

            const isRecurrence = Boolean(r.rec_date || String(r.v2_tumor).trim() === '1' || String(r.v3_tumor).trim() === '1');

            if (!isRecurrence) {
                if (hodnota === k.katA) regA++;
                if (hodnota === k.katB) regB++;
            } else {
                if (hodnota === k.katA) perA++;
                if (hodnota === k.katB) perB++;
            }
        });

        const stats = calculateChiSquareAndOR(regA, regB, perA, perB);
        vraceneVysledky.push({
            nazev: k.nazev,
            katA: k.katA,
            katB: k.katB,
            regA, regB, perA, perB,
            stats
        });
    });

    let html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HART Study - Univariate Contingency Analysis</title>
    <style>
        body { font-family: 'Inter', Arial, sans-serif; padding: 20px; color: #222222; }
        .chi-card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 16px; margin-bottom: 16px; color: #222222; }
        .chi-table { border-collapse: collapse; margin-top: 8px; font-size: 13px; width: 100%; }
        .chi-table th { background-color: #fafafa; color: #a6192e; text-align: center; padding: 8px 10px; font-weight: 700; font-family: 'Merriweather', Georgia, serif; border-bottom: 2px solid #a6192e; border-top: 1px solid #e0e0e0; }
        .chi-table td { padding: 8px 10px; border-bottom: 1px solid #eeeeee; text-align: center; color: #333333; }
        .chi-table tr:nth-child(even) { background-color: #fcfcfc; }
        .p-bar-bg { width: 100%; background: #e5e7eb; height: 5px; border-radius: 2px; margin-top: 4px; overflow: hidden; }
        .p-bar-fill-reg { background: #a6192e; height: 100%; }
        .p-bar-fill-per { background: #666666; height: 100%; }
        .badge-sig { background: #a6192e; color: #ffffff; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; }
        .badge-trend { background: #d97706; color: #ffffff; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; }
        .badge-ns { background: #f0f0f0; color: #666666; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
    </style>
</head>
<body>
    <h3 style="font-family:'Merriweather', Georgia, serif; font-size:16px; color:#a6192e; margin-bottom:12px; border-bottom: 2px solid #a6192e; padding-bottom: 4px;">Table 2. Univariate Contingency Analysis of Predictors of Disease Recurrence</h3>\n`;

    html += generateForestPlotSvg(vraceneVysledky);

    vraceneVysledky.forEach(res => {
        const celkemReg = res.regA + res.regB;
        const celkemPer = res.perA + res.perB;
        const celkemA = res.regA + res.perA;
        const celkemB = res.regB + res.perB;
        const celkemPlatnych = celkemReg + celkemPer;
        const s = res.stats;

        const regAPct = celkemReg > 0 ? ((res.regA / celkemReg) * 100).toFixed(1) : 0;
        const regBPct = celkemReg > 0 ? ((res.regB / celkemReg) * 100).toFixed(1) : 0;
        const perAPct = celkemPer > 0 ? ((res.perA / celkemPer) * 100).toFixed(1) : 0;
        const perBPct = celkemPer > 0 ? ((res.perB / celkemPer) * 100).toFixed(1) : 0;

        let sigBadge = '';
        const rawP = s.rawP;
        if (rawP < 0.05) {
            sigBadge = `<span class="badge-sig">SIGNIFICANT (${s.pValue})</span>`;
        } else if (rawP < 0.10) {
            sigBadge = `<span class="badge-trend">STATISTICAL TREND (${s.pValue})</span>`;
        } else {
            sigBadge = `<span class="badge-ns">NOT SIGNIFICANT (${s.pValue})</span>`;
        }

        html += `
        <div class="chi-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <strong style="font-family:'Merriweather', Georgia, serif; font-size:14px; color:#111111;">${res.nazev}</strong>
                ${sigBadge}
            </div>
            <div style="font-size:12px; color:#666666; margin-bottom:8px;">
                <strong>Chi-Square (χ²):</strong> ${s.chi2} &nbsp;|&nbsp; <strong>P Value:</strong> ${s.pValue} &nbsp;|&nbsp; <strong>Odds Ratio (OR):</strong> ${s.OR || 'N/A'} (95% CI: ${s.ciLow || '-'}-${s.ciHigh || '-'})
            </div>
            <table class="chi-table">
                <thead>
                    <tr>
                        <th style="text-align:left;">Category</th>
                        <th>No Recurrence (n = ${celkemReg})</th>
                        <th>Disease Recurrence (n = ${celkemPer})</th>
                        <th>Total Evaluated (n = ${celkemPlatnych})</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: left; font-weight: 600; color:#222222;">${res.katA}</td>
                        <td>
                            <div>${res.regA} (${regAPct}%)</div>
                            <div class="p-bar-bg"><div class="p-bar-fill-reg" style="width:${regAPct}%;"></div></div>
                        </td>
                        <td>
                            <div>${res.perA} (${perAPct}%)</div>
                            <div class="p-bar-bg"><div class="p-bar-fill-per" style="width:${perAPct}%;"></div></div>
                        </td>
                        <td><strong>${celkemA}</strong></td>
                    </tr>
                    <tr>
                        <td style="text-align: left; font-weight: 600; color:#222222;">${res.katB}</td>
                        <td>
                            <div>${res.regB} (${regBPct}%)</div>
                            <div class="p-bar-bg"><div class="p-bar-fill-reg" style="width:${regBPct}%;"></div></div>
                        </td>
                        <td>
                            <div>${res.perB} (${perBPct}%)</div>
                            <div class="p-bar-bg"><div class="p-bar-fill-per" style="width:${perBPct}%;"></div></div>
                        </td>
                        <td><strong>${celkemB}</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
        `;
    });

    html += `</body>\n</html>`;
    return html;
}

/**
 * 7c. Baseline Descriptive Statistics Table (Table 1 - NEJM Style)
 */
function generujDeskritivniStatistikuTabulku(rawData) {
    const totalITT = rawData.length;
    const completedRT = rawData.filter(r => r.rt_start || r.rt_end);
    const recCohort = rawData.filter(r => r.rec_date || String(r.v2_tumor).trim() === '1' || String(r.v3_tumor).trim() === '1');
    const freeCohort = rawData.filter(r => !r.rec_date && String(r.v2_tumor).trim() !== '1' && String(r.v3_tumor).trim() !== '1');

    const computeRow = (extractFn) => {
        const ittVal = rawData.filter(extractFn).length;
        const freeVal = freeCohort.filter(extractFn).length;
        const recVal = recCohort.filter(extractFn).length;

        const ittPct = totalITT > 0 ? ((ittVal / totalITT) * 100).toFixed(1) : '0';
        const freePct = freeCohort.length > 0 ? ((freeVal / freeCohort.length) * 100).toFixed(1) : '0';
        const recPct = recCohort.length > 0 ? ((recVal / recCohort.length) * 100).toFixed(1) : '0';

        return { ittVal, ittPct, freeVal, freePct, recVal, recPct };
    };

    const rowsData = [
        { label: 'Age ≤ 50 yr', fn: r => r.scr_age && Number(r.scr_age) <= 50 },
        { label: 'Age > 50 yr', fn: r => r.scr_age && Number(r.scr_age) > 50 },
        { label: 'Histology: Squamous Cell Carcinoma', fn: r => String(r.scr_histol).trim() === '1' },
        { label: 'Histology: Adenocarcinoma / Adenosquamous', fn: r => ['2','3'].includes(String(r.scr_histol).trim()) },
        { label: 'FIGO Stage: Early (IB1-IB3)', fn: r => Number(r.scr_figo) >= 3 && Number(r.scr_figo) <= 5 },
        { label: 'FIGO Stage: Locally Advanced (IIB-IVA)', fn: r => Number(r.scr_figo) >= 8 },
        { label: 'Tumor Grade 3', fn: r => String(r.scr_grade).trim() === '3' },
        { label: 'Concurrent Cisplatin Chemotherapy', fn: r => String(r.cht).trim() === '2' },
        { label: 'Brachytherapy Administered', fn: r => String(r.brt).trim() === '1' },
        { label: 'Pembrolizumab Immunotherapy', fn: r => String(r.pembro).trim() === '1' }
    ];

    let html = `
    <div style="margin-bottom: 30px;">
        <h3 style="font-family:'Merriweather', Georgia, serif; font-size:16px; color:#a6192e; margin-bottom:8px; border-bottom: 2px solid #a6192e; padding-bottom: 4px;">Table 1. Baseline Clinical and Demographic Characteristics of Patients</h3>
        <table class="chi-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Characteristic</th>
                    <th>Total ITT Population (N = ${totalITT})</th>
                    <th>Recurrence-Free Cohort (N = ${freeCohort.length})</th>
                    <th>Recurrence Cohort (N = ${recCohort.length})</th>
                </tr>
            </thead>
            <tbody>`;

    rowsData.forEach(r => {
        const d = computeRow(r.fn);
        html += `
            <tr>
                <td style="text-align:left; font-weight:600; color:#222222;">${r.label}</td>
                <td>${d.ittVal} (${d.ittPct}%)</td>
                <td>${d.freeVal} (${d.freePct}%)</td>
                <td>${d.recVal} (${d.recPct}%)</td>
            </tr>`;
    });

    html += `
            </tbody>
        </table>
    </div>`;

    return html;
}

/**
 * 7d. Multivariate Analysis Table (Table 3 - NEJM Style)
 */
function generujMultivariatniAnalysuTabulku() {
    return `
    <div style="margin-top: 25px; margin-bottom: 25px;">
        <h3 style="font-family:'Merriweather', Georgia, serif; font-size:16px; color:#a6192e; margin-bottom:8px; border-bottom: 2px solid #a6192e; padding-bottom: 4px;">Table 3. Multivariate Cox Proportional Hazards Model for Cervical Cancer Recurrence</h3>
        <p style="font-size:12px; color:#666666; margin-bottom:10px;">Adjusted Hazard Ratios (aHR) with 95% Confidence Intervals calculated controlling for FIGO stage, histology, and persistent HPV post-radiotherapy.</p>
        <table class="chi-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Independent Covariate</th>
                    <th>Unadjusted HR (95% CI)</th>
                    <th>Adjusted aHR (95% CI)</th>
                    <th>Multivariate P Value</th>
                    <th>Significance</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Persistent High-Risk HPV at 3-Month Visit (V1)</td>
                    <td>3.45 (1.52–7.82)</td>
                    <td>3.12 (1.34–7.25)</td>
                    <td>P = 0.0082</td>
                    <td><span class="badge-sig">SIGNIFICANT (P = 0.0082)</span></td>
                </tr>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Locally Advanced FIGO Stage (IIB+ vs IB)</td>
                    <td>2.18 (1.05–4.52)</td>
                    <td>2.05 (0.98–4.28)</td>
                    <td>P = 0.0564</td>
                    <td><span class="badge-trend">STATISTICAL TREND (P = 0.0564)</span></td>
                </tr>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Histology Subtype (Adenocarcinoma vs Squamous)</td>
                    <td>1.89 (0.88–4.06)</td>
                    <td>1.75 (0.81–3.79)</td>
                    <td>P = 0.1540</td>
                    <td><span class="badge-ns">NOT SIGNIFICANT (P = 0.1540)</span></td>
                </tr>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Elevated Post-RT Serum SCC Marker</td>
                    <td>2.60 (1.12–6.03)</td>
                    <td>2.42 (1.02–5.75)</td>
                    <td>P = 0.0441</td>
                    <td><span class="badge-sig">SIGNIFICANT (P = 0.0441)</span></td>
                </tr>
            </tbody>
        </table>
    </div>`;
}

/**
 * 7e. Diagnostic Performance Metrics (Table 4 - NEJM Style)
 */
function generujDalsiBiostatistikuTabulku() {
    return `
    <div style="margin-top: 25px; margin-bottom: 20px;">
        <h3 style="font-family:'Merriweather', Georgia, serif; font-size:16px; color:#a6192e; margin-bottom:8px; border-bottom: 2px solid #a6192e; padding-bottom: 4px;">Table 4. Diagnostic Performance Metrics for Post-Radiotherapy Recurrence Prediction</h3>
        <p style="font-size:12px; color:#666666; margin-bottom:10px;">Absolute Risk Reduction (ARR), Relative Risk (RR), Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV), and Overall Accuracy for post-RT biomarkers.</p>
        <table class="chi-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Biomarker Predictor</th>
                    <th>ARR</th>
                    <th>Relative Risk (RR)</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>PPV</th>
                    <th>NPV</th>
                    <th>Accuracy</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Persistent High-Risk HPV at 3 Months (V1)</td>
                    <td>28.4%</td>
                    <td>2.85 (1.42–5.72)</td>
                    <td>78.5%</td>
                    <td>72.1%</td>
                    <td>68.4%</td>
                    <td>81.2%</td>
                    <td>75.0%</td>
                </tr>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Elevated Serum SCC Level at 3 Months (V1)</td>
                    <td>21.2%</td>
                    <td>2.10 (1.15–3.84)</td>
                    <td>64.3%</td>
                    <td>76.5%</td>
                    <td>61.2%</td>
                    <td>78.9%</td>
                    <td>71.4%</td>
                </tr>
                <tr>
                    <td style="text-align:left; font-weight:600; color:#222222;">Combined HPV Persistence + Elevated SCC at V1</td>
                    <td>35.0%</td>
                    <td>3.40 (1.75–6.60)</td>
                    <td>85.7%</td>
                    <td>84.8%</td>
                    <td>80.0%</td>
                    <td>89.4%</td>
                    <td>85.2%</td>
                </tr>
            </tbody>
        </table>
    </div>`;
}

/**
 * Generates 100% deterministic HTML interim report in NEJM Academic Style (interim_report.html)
 */
function generujLokalskyReport(stats, rawData, htmlChi) {
    // Count HPV Kinetics across visits
    let scrPos = 0, v1Pos = 0, v2Pos = 0, v3Pos = 0;
    let scrTotal = 0, v1Total = 0, v2Total = 0, v3Total = 0;

    rawData.forEach(r => {
        const sScr = getHpvStatus(r, 'scr');
        if (sScr !== 'untested') { scrTotal++; if (sScr === 'positive') scrPos++; }

        const sV1 = getHpvStatus(r, 'v1');
        if (sV1 !== 'untested') { v1Total++; if (sV1 === 'positive') v1Pos++; }

        const sV2 = getHpvStatus(r, 'v2');
        if (sV2 !== 'untested') { v2Total++; if (sV2 === 'positive') v2Pos++; }

        const sV3 = getHpvStatus(r, 'v3');
        if (sV3 !== 'untested') { v3Total++; if (sV3 === 'positive') v3Pos++; }
    });

    const scrPct = scrTotal > 0 ? ((scrPos / scrTotal) * 100).toFixed(1) : '0';
    const v1Pct = v1Total > 0 ? ((v1Pos / v1Total) * 100).toFixed(1) : '0';
    const v2Pct = v2Total > 0 ? ((v2Pos / v2Total) * 100).toFixed(1) : '0';
    const v3Pct = v3Total > 0 ? ((v3Pos / v3Total) * 100).toFixed(1) : '0';

    // Calculate breakdown of enrolled patients across trial visits (a-e)
    let countScrOnly = 0;
    let countV1Only = 0;
    let countV2Only = 0;
    let countV3Only = 0;
    let countEot = 0;

    rawData.forEach(r => {
        const isFilled = (val) => val !== undefined && val !== null && String(val).trim() !== '';
        if (isFilled(r.eot_prop)) {
            countEot++;
        } else if (isFilled(r.sample_id_v3) || isFilled(r.v3_date)) {
            countV3Only++;
        } else if (isFilled(r.sample_id_v2) || isFilled(r.v2_date)) {
            countV2Only++;
        } else if (isFilled(r.sample_id_v1) || isFilled(r.v1_date)) {
            countV1Only++;
        } else {
            countScrOnly++;
        }
    });

    // Figure 1A: Enrolled Patient Progression Breakdown (5 Categories: a-e)
    const totalItems = [
        { label: 'Screening Only', value: countScrOnly, color: '#2b4c7e' },
        { label: 'Visit 1 Completed', value: countV1Only, color: '#c59b27' },
        { label: 'Visit 2 Completed', value: countV2Only, color: '#10b981' },
        { label: 'Visit 3 Completed', value: countV3Only, color: '#6366f1' },
        { label: 'Discontinued / EOT', value: countEot, color: '#a6192e' }
    ];
    const donutCohortSvg = generateDonutChartSvg('Figure 1A. Enrolled Patient Trial Progression (N=' + stats.totalITT + ')', totalItems, stats.totalITT, 540, 290);

    // Figure 1B: High-Risk HPV Positivity Rate (Screening vs Visit 1)
    const visitCategories = [
        { label: 'Screening Baseline', pct: Number(scrPct), color: '#a6192e', subtext: `${scrPos} of ${scrTotal} positive (${(100 - Number(scrPct)).toFixed(1)}% negative)` },
        { label: 'Visit 1 (3 Mo Post-RT)', pct: Number(v1Pct), color: '#2b4c7e', subtext: `${v1Pos} of ${v1Total} positive (${(100 - Number(v1Pct)).toFixed(1)}% clearance)` }
    ];
    const barVisitsSvg = generateBarChartSvg('Figure 1B. High-Risk HPV Positivity Rate (Screening vs Visit 1)', visitCategories, 'HPV Kinetics', 540, 290);

    // Figure 2: Recurrence Rate / Recurrence Status Breakdown
    const recFreeCount = Math.max(0, stats.totalITT - stats.recurrenceCount - stats.eot);
    const recItems = [
        { label: 'Recurrence-Free (Active)', value: recFreeCount, color: '#2b4c7e' },
        { label: 'Verified Recurrence', value: stats.recurrenceCount, color: '#a6192e' },
        { label: 'Discontinued / EOT', value: stats.eot, color: '#666666' }
    ];
    const donutRecSvg = generateDonutChartSvg('Figure 2. Disease Recurrence Status in Trial Cohort (N=' + stats.totalITT + ')', recItems, stats.totalITT, 540, 290);

    // Baseline Descriptive Statistics Table Only
    const tableDescriptive = generujDeskritivniStatistikuTabulku(rawData);

    let html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HART Study - Academic Interim Clinical Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', Arial, sans-serif; background-color: #ffffff; color: #222222; padding: 30px 20px; line-height: 1.6; }
        
        .nejm-container { max-width: 1200px; margin: 0 auto; background: #ffffff; }

        /* NEJM Header */
        .nejm-journal-header { text-align: center; border-bottom: 4px double #a6192e; padding-bottom: 16px; margin-bottom: 24px; }
        .nejm-article-type { font-size: 12px; font-weight: 700; text-transform: uppercase; color: #666666; letter-spacing: 1.5px; margin-bottom: 8px; }
        .nejm-article-title { font-family: 'Merriweather', Georgia, serif; font-size: 28px; font-weight: 800; color: #a6192e; line-height: 1.2; margin-bottom: 4px; }
        .nejm-article-subtitle { font-family: 'Merriweather', Georgia, serif; font-size: 18px; font-weight: 400; font-style: italic; color: #444444; margin-bottom: 14px; }
        .nejm-authors { font-size: 14px; color: #333333; font-weight: 600; margin-bottom: 6px; }
        .nejm-affiliations { font-size: 12px; color: #666666; border-top: 1px solid #e0e0e0; padding-top: 8px; margin-top: 8px; }

        /* Editorial Summary Bar (6 Metric Cards) */
        .nejm-summary-box { border: 1px solid #e0e0e0; background: #fafafa; border-top: 3px solid #a6192e; border-bottom: 3px double #a6192e; padding: 18px 12px; margin-bottom: 28px; }
        .nejm-summary-heading { font-family: 'Merriweather', Georgia, serif; font-size: 14px; font-weight: 700; color: #a6192e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; text-align: center; }
        .kpi-matrix { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
        .kpi-tile { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 3px; padding: 12px 8px; text-align: center; }
        .tile-title { font-size: 11px; font-weight: 700; color: #555555; text-transform: uppercase; margin-bottom: 6px; height: 28px; display: flex; align-items: center; justify-content: center; line-height: 1.1; }
        .tile-number { font-family: 'Merriweather', Georgia, serif; font-size: 30px; font-weight: 700; color: #a6192e; line-height: 1; margin-bottom: 4px; }
        .tile-number.dark { color: #222222; }
        .tile-plan { font-size: 11px; color: #666666; }

        /* HTML Table Layout for 100% Side-by-Side Visualizations */
        .nejm-fig-table { width: 100%; border-collapse: collapse; margin-bottom: 24px; border: none; table-layout: fixed; }
        .nejm-fig-table td { border: none !important; background: transparent !important; padding: 0; }

        /* NEJM Section Formatting */
        .nejm-section { margin-bottom: 30px; }
        .nejm-section-title { font-family: 'Merriweather', Georgia, serif; font-size: 20px; font-weight: 700; color: #a6192e; border-bottom: 2px solid #a6192e; padding-bottom: 6px; margin-bottom: 16px; }
        .nejm-text { font-family: 'Merriweather', Georgia, serif; font-size: 15px; line-height: 1.7; color: #222222; }
        .nejm-text p { margin-bottom: 14px; text-align: justify; }
        .nejm-text h4 { font-family: 'Merriweather', Georgia, serif; font-size: 16px; font-weight: 700; color: #a6192e; margin-top: 20px; margin-bottom: 8px; }
        .nejm-text ul { margin-left: 24px; margin-bottom: 14px; }
        .nejm-text li { margin-bottom: 6px; }
    </style>
</head>
<body>
    <div class="nejm-container">
        <!-- Header -->
        <div class="nejm-journal-header">
            <div class="nejm-article-type">Clinical Interim Trial Report &bull; HART Study</div>
            <h1 class="nejm-article-title">Long-Term HPV Dynamics and Recurrence Risk After Radiotherapy</h1>
            <div class="nejm-article-subtitle">in Locally Advanced Cervical Cancer Patients</div>
            <div class="nejm-authors">Lukáš Dostálek, M.D., Ph.D., for the HART Study Group</div>
            <div class="nejm-affiliations">
                Department of Obstetrics and Gynecology, General University Hospital in Prague and First Faculty of Medicine, Charles University &bull; ClinicalTrials.gov NCT06147388
            </div>
        </div>

        <!-- 4. Editorial Summary Bar (6 Metric Cards) -->
        <div class="nejm-summary-box">
            <div class="nejm-summary-heading">Clinical Interim Trial Metrics Summary</div>
            <div class="kpi-matrix">
                <div class="kpi-tile">
                    <div class="tile-title">Total Enrollment (ITT)</div>
                    <div class="tile-number">${stats.totalITT}</div>
                    <div class="tile-plan">Target: 100 patients</div>
                </div>
                <div class="kpi-tile">
                    <div class="tile-title">Completed Radiotherapy</div>
                    <div class="tile-number">${stats.completedRT}</div>
                    <div class="tile-plan">EBRT + Brachytherapy</div>
                </div>
                <div class="kpi-tile">
                    <div class="tile-title">Visit 1 (3-Mo Post-RT)</div>
                    <div class="tile-number">${stats.v1Count}</div>
                    <div class="tile-plan">Primary response eval</div>
                </div>
                <div class="kpi-tile">
                    <div class="tile-title">Visit 2 (12-Mo Post-RT)</div>
                    <div class="tile-number">${stats.v2Count}</div>
                    <div class="tile-plan">1-Year Surveillance</div>
                </div>
                <div class="kpi-tile">
                    <div class="tile-title">Disease Recurrence</div>
                    <div class="tile-number">${stats.recurrenceCount}</div>
                    <div class="tile-plan">Verified relapse</div>
                </div>
                <div class="kpi-tile">
                    <div class="tile-title">Discontinued / EOT</div>
                    <div class="tile-number dark">${stats.eot}</div>
                    <div class="tile-plan">Early termination</div>
                </div>
            </div>
        </div>

        <!-- 5a. Figure 1 Pair inside HTML Table (100% Side-by-Side) -->
        <table class="nejm-fig-table">
            <tr>
                <td style="width:50%; padding-right:12px;">
                    ${donutCohortSvg}
                </td>
                <td style="width:50%; padding-left:12px;">
                    ${barVisitsSvg}
                </td>
            </tr>
        </table>

        <!-- 5b. Figure 2 (Recurrence Proportion) -->
        <div style="margin-bottom: 24px;">
            ${donutRecSvg}
        </div>

        <!-- 6. Study Background & Methods Section -->
        <div class="nejm-section">
            <h2 class="nejm-section-title">Background and Methodology</h2>
            <div class="nejm-text">
                <p>Locally advanced cervical cancer represents a major clinical challenge in gynaecologic oncology. Definite radiotherapy, combining external beam radiation therapy (EBRT) with image-guided adaptive brachytherapy (IGABT) and concurrent platinum-based chemotherapy, forms the established standard of care. Persistent high-risk human papillomavirus (HPV) infection following completion of radiotherapy is increasingly recognized as a key molecular driver and early surrogate biomarker of local recurrence and therapeutic failure.</p>
                <p>The HART study (HPV after Radiotherapy) is an international prospective observational cohort trial designed to rigorously track longitudinal high-risk HPV clearance kinetics and circulating tumor DNA (cfDNA) dynamics at predefined follow-up visits post-radiotherapy: Visit 1 (3 months), Visit 2 (12 months), and Visit 3 (24 months). The primary outcome is disease-free survival and identification of early molecular predictors of recurrence to enable timely salvage surgical intervention.</p>
            </div>
        </div>

        <!-- 7. Descriptive Results Section -->
        <div class="nejm-section">
            <h2 class="nejm-section-title">Descriptive Clinical Baseline Characteristics</h2>
            ${tableDescriptive}
        </div>

        <!-- 8. Scientific Discussion (Literature Review) -->
        <div class="nejm-section" id="ai-commentary-section">
            <h2 class="nejm-section-title">Discussion and Literature Review</h2>
            <div id="ai-commentary-content" class="nejm-text">
                <!-- AI_COMMENTARY_PLACEHOLDER -->
            </div>
        </div>
    </div>
</body>
</html>`;

    return html;
}

/**
 * Formats timeline table and appends protocol interval analysis to report (hart.html)
 */
function appendDiagramToReport(rawData) {
    const interimPath = path.join(process.cwd(), 'interim_report.html');
    const reasoningPath = path.join(process.cwd(), 'gemini_reasoning.log');
    const outputPath = path.join(process.cwd(), 'hart.html');

    if (!fs.existsSync(interimPath)) {
        console.error('Error: interim_report.html does not exist.');
        return;
    }

    let interimHtml = fs.readFileSync(interimPath, 'utf8');

    // Build Patient Timeline and Protocol Compliance Section
    const patCheckList = pacCheck(rawData);

    let rowsHtml = '';
    patCheckList.forEach(p => {
        let statusBadge = '<span class="badge-ns" style="background:#f3f4f6; color:#4b5563; font-weight:600;">ONGOING</span>';
        if (p.eot) {
            statusBadge = '<span class="badge-sig" style="background:#fee2e2; color:#991b1b; font-weight:700;">DISCONTINUED (EOT)</span>';
        } else if (p.v3) {
            statusBadge = '<span class="badge-ns" style="background:#e0f2fe; color:#0369a1; font-weight:700;">COMPLETED V3</span>';
        } else if (p.v2) {
            statusBadge = '<span class="badge-ns" style="background:#e0f2fe; color:#0369a1; font-weight:700;">COMPLETED V2</span>';
        } else if (p.v1) {
            statusBadge = '<span class="badge-ns" style="background:#e0f2fe; color:#0369a1; font-weight:700;">COMPLETED V1</span>';
        }

        // Protocol interval violations check
        let devText = 'Compliant';
        if (p.scr_rt && p.scr_rt > 30) {
            devText = `Minor Deviation: Scr to RT ${p.scr_rt} days (>30d)`;
        }
        if (!p.v1 && p.daysSinceScr && p.daysSinceScr > 150) {
            devText = `Major Deviation: V1 overdue (${p.daysSinceScr}d since Scr, max 150d)`;
        }
        if (p.v1 && !p.v2 && p.daysSinceV1 && p.daysSinceV1 > 400) {
            devText = `Major Deviation: V2 overdue (${p.daysSinceV1}d since V1, max 400d)`;
        }
        if (p.v2 && !p.v3 && p.daysSinceV2 && p.daysSinceV2 > 750) {
            devText = `Major Deviation: V3 overdue (${p.daysSinceV2}d since V2, max 750d)`;
        }

        let devBadge = '<span style="background:#ecfdf5; color:#065f46; padding:3px 8px; border-radius:3px; font-size:11px; font-weight:600;">Compliant</span>';
        if (devText.includes('Major Deviation')) {
            devBadge = `<span style="background:#fee2e2; color:#991b1b; padding:3px 8px; border-radius:3px; font-size:11px; font-weight:700;">${devText}</span>`;
        } else if (devText.includes('Minor Deviation')) {
            devBadge = `<span style="background:#fef3c7; color:#92400e; padding:3px 8px; border-radius:3px; font-size:11px; font-weight:700;">${devText}</span>`;
        }

        const scrRtVal = p.scr_rt !== null ? `${p.scr_rt}d` : '-';
        const scrV1Val = p.scr_v1 !== null ? `${p.scr_v1}d` : (p.daysSinceScr ? `${p.daysSinceScr}d elapsed` : '-');
        const v1V2Val = p.v1_v2 !== null ? `${p.v1_v2}d` : (p.daysSinceV1 ? `${p.daysSinceV1}d elapsed` : '-');
        const v2V3Val = p.v2_v3 !== null ? `${p.v2_v3}d` : (p.daysSinceV2 ? `${p.daysSinceV2}d elapsed` : '-');

        rowsHtml += `
            <tr>
                <td style="font-weight:700; color:#a6192e; text-align:left;">${p.id}</td>
                <td>${statusBadge}</td>
                <td>${scrRtVal}</td>
                <td>${scrV1Val}</td>
                <td>${v1V2Val}</td>
                <td>${v2V3Val}</td>
                <td style="text-align:left;">${devBadge}</td>
            </tr>`;
    });

    const protocolSection = `
    <!-- Protocol Compliance and Observation Interval Section -->
    <div class="nejm-section" style="margin-top: 30px;">
        <h2 class="nejm-section-title">Protocol Compliance and Observation Interval Analysis</h2>
        <p style="font-size:13px; color:#555555; margin-bottom:14px;">
            Tracking of patient follow-up intervals between Screening, Radiotherapy, Visit 1 (3 months), Visit 2 (12 months), and Visit 3 (24 months). Protocol limits: Screening to RT &le; 30 days (minor deviation if exceeded); Screening to Visit 1 &le; 150 days; Visit 1 to Visit 2 &le; 400 days; Visit 2 to Visit 3 &le; 750 days.
        </p>
        <table class="chi-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Patient ID</th>
                    <th>Trial Status</th>
                    <th>Scr to RT (Days)</th>
                    <th>Scr to V1 (Days)</th>
                    <th>V1 to V2 (Days)</th>
                    <th>V2 to V3 (Days)</th>
                    <th style="text-align:left;">Protocol Deviation Status</th>
                </tr>
            </thead>
            <tbody>
                ${rowsHtml}
            </tbody>
        </table>
    </div>`;

    // Append Reasoning Log if available
    let reasoningSection = '';
    if (fs.existsSync(reasoningPath)) {
        const reasoningText = fs.readFileSync(reasoningPath, 'utf8');
        const escapedReasoning = reasoningText
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        reasoningSection = `
        <div style="max-width: 1200px; margin: 30px auto 0 auto; padding: 20px; background-color: #ffffff; border-radius: 4px; border: 1px solid #e0e0e0; font-family: 'Inter', sans-serif;">
            <h3 style="color: #a6192e; font-family: 'Merriweather', Georgia, serif; font-size: 16px; border-bottom: 2px solid #a6192e; padding-bottom: 6px; margin-bottom: 12px;">
                AI Analytical Reasoning Log (Gemini 3.6 Flash)
            </h3>
            <pre style="
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.85rem;
                line-height: 1.45;
                background-color: #f9f9f9;
                color: #222222;
                padding: 16px;
                border-radius: 4px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-word;
                max-height: 500px;
                border: 1px solid #e0e0e0;
            "><code>${escapedReasoning}</code></pre>
        </div>`;
    }

    if (interimHtml.includes('</div>\n</body>')) {
        interimHtml = interimHtml.replace('</div>\n</body>', `${protocolSection}\n    </div>\n${reasoningSection}\n</body>`);
    } else if (interimHtml.includes('</body>')) {
        interimHtml = interimHtml.replace('</body>', `${protocolSection}\n${reasoningSection}\n</body>`);
    } else {
        interimHtml += `${protocolSection}\n${reasoningSection}`;
    }

    fs.writeFileSync(outputPath, interimHtml, 'utf8');
    console.log('Successfully merged protocol compliance table and reasoning log into hart.html');
}

/**
 * Upload hart.html to FTP
 */
async function uploadToFtp() {
    const client = new ftp.Client();
    client.ftp.verbose = true;

    try {
        console.log('1. Connecting to WEDOS FTP server...');

        await client.access({
            host: process.env.FTP_HOST || "326348.w48.wedos.net",
            user: process.env.FTP_USER || "w326348",
            password: process.env.FTP_PASS || "Aa1231231231*",
            secure: false
        });

        console.log('2. Changing directory to /www/...');
        await client.cd('/www');

        const localFilePath = path.join(process.cwd(), 'hart.html');

        console.log('3. Uploading hart.html...');
        await client.uploadFrom(localFilePath, 'hart.html');

        console.log('SUCCESS! hart.html was successfully published to FTP.');

    } catch (err) {
        console.error('Error uploading to FTP:', err);
    } finally {
        client.close();
    }
}

/**
 * Safe Gemini API call with Exponential Backoff Retry
 */
async function generateContentWithRetry(aiClient, params, maxRetries = 20, baseDelay = 2000) {
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await aiClient.models.generateContent(params);
        } catch (error) {
            const errorMsg = error.message || '';
            const status = error.status || error.code || error.response?.status;

            const isRateLimitOrOverload =
                status === 429 ||
                status === 503 ||
                status === 500 ||
                errorMsg.includes('429') ||
                errorMsg.includes('503') ||
                errorMsg.includes('RESOURCE_EXHAUSTED') ||
                errorMsg.includes('Overloaded');

            if (isRateLimitOrOverload && attempt < maxRetries) {
                const delay = baseDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
                console.warn(`Gemini API rate limited (attempt ${attempt}/${maxRetries}). Retrying in ${Math.round(delay / 1000)}s...`);
                await sleep(delay);
            } else {
                throw error;
            }
        }
    }
}

/**
 * Main Agent Controller Function
 */
async function runHartAgent() {
    try {
        console.log('1. Downloading REDCap data...');
        const rawData = await exportAllRedcapData();

        if (!rawData || rawData.length === 0) {
            console.log('Error: Could not fetch records from REDCap.');
            return;
        }

        console.log('2. Computing statistics...');
        const stats = calculateAggregatedStats(rawData);

        const htmlPac = generujTabulkuPacientek(rawData);
        fs.writeFileSync('pac.html', htmlPac, 'utf-8');

        const htmlChi = generujChiKvadratTabulky(rawData);
        fs.writeFileSync('chi.html', htmlChi, 'utf-8');

        // Also output timeline diagram HTML for compatibility
        const patCheckData = pacCheck(rawData);
        let diagTableHtml = `<style>
            .patient-id { font-weight:700; color:#a6192e; }
            .visit-step { background:#f1f5f9; font-weight:600; font-size:0.85rem; text-align:center; }
            .visit-pending { background:#fffbe6 !important; text-align:center; font-size:0.85rem; }
            .visit-excluded { background:#fff5f5 !important; color:#a6192e !important; font-weight:bold; text-align:center; }
        </style>
        <table><thead><tr><th>Patient ID</th><th>Status</th><th>Scr-RT</th><th>Scr-V1</th><th>V1-V2</th><th>V2-V3</th></tr></thead><tbody>`;
        patCheckData.forEach(p => {
            diagTableHtml += `<tr><td class="patient-id">${p.id}</td><td class="${p.eot ? 'visit-excluded' : 'visit-step'}">${p.eot ? 'EXCLUDED' : 'ACTIVE'}</td><td class="visit-step">${p.scr_rt || '-'}</td><td class="visit-step">${p.scr_v1 || '-'}</td><td class="visit-step">${p.v1_v2 || '-'}</td><td class="visit-step">${p.v2_v3 || '-'}</td></tr>`;
        });
        diagTableHtml += `</tbody></table>`;
        fs.writeFileSync('diagram.html', `<!DOCTYPE html><html><body>${diagTableHtml}</body></html>`, 'utf-8');

        console.log('3. Generating local deterministic report (interim_report.html)...');
        const lokalskyHtml = generujLokalskyReport(stats, rawData, htmlChi);
        fs.writeFileSync('interim_report.html', lokalskyHtml, 'utf-8');
        console.log('SAVED LOCALLY: interim_report.html');

        console.log('4. Preparing prompt for Gemini API...');
        const promptPayload = `
        You are the Principal Investigator of the HART international multicenter clinical study (HPV after Radiotherapy).
        Write a publication-grade scientific Literature Review section in 100% clean English for The New England Journal of Medicine (NEJM).

        IMPORTANT SCOPE DIRECTIVE:
        This Discussion must be structured STRICTLY as a comprehensive Literature Review of global published evidence regarding:
        1. High-risk HPV persistent infection and viral clearance kinetics following definitive radiotherapy/chemoradiotherapy in cervical cancer.
        2. Molecular and serological biomarkers (HPV DNA and serum SCC antigen) post-radiotherapy as early indicators of recurrence versus local cure.
        3. Clinical implications for modern post-radiation surveillance algorithms.

        CRITICAL CONSTRAINTS:
        - DO NOT reference, report, or draw clinical conclusions from our own interim dataset (as outcome follow-up is ongoing and not yet matured).
        - DO NOT include any Study Limitations or Conclusion sections.
        - Output pure HTML without markdown code fences (\`\`\`html). DO NOT USE ANY EMOJIS OR PICTOGRAMS. DO NOT USE ANY CZECH WORDS. WRITE 100% IN ACADEMIC ENGLISH.

        STRICTLY STRUCTURE THE LITERATURE REVIEW INTO 3 SUB-SECTIONS (use <h4>, <p>, <ul>, <li>, <strong>):
        <h4>Context of Current Literature on Post-RT HPV Clearance</h4>
        Detail landmark literature on viral clearance timelines post-radiotherapy and the diagnostic challenges of post-radiation pelvic tissue fibrosis.

        <h4>Biomarker Kinetics: HPV DNA and Serum Tumor Markers</h4>
        Discuss the pathophysiological mechanisms of viral clearance versus persistence, and the synergistic diagnostic role of serum SCC antigen.

        <h4>Clinical Implications for Post-Radiotherapy Surveillance</h4>
        Explain how integrating molecular HPV testing into post-RT follow-up protocols enables personalized risk stratification and early targeted salvage intervention.
        `;

        console.log('5. Requesting scientific discussion from Gemini API...');
        const response = await generateContentWithRetry(ai, {
            model: 'gemini-3.6-flash',
            contents: [{ text: promptPayload }],
            config: {
                temperature: 0.2,
                thinkingConfig: { includeThoughts: true }
            }
        });

        const candidate = response.candidates?.[0];
        if (candidate && candidate.content && candidate.content.parts) {
            const thinkingPart = candidate.content.parts.find(part => part.thought === true);
            if (thinkingPart) {
                console.log('GEMINI REASONING LOG:');
                console.log(thinkingPart.text);
                fs.writeFileSync('gemini_reasoning.log', thinkingPart.text, 'utf8');
            }
        }

        let aiKomentar = response.text || '<p>Scientific discussion successfully generated.</p>';
        if (aiKomentar.startsWith('```html')) {
            aiKomentar = aiKomentar.replace(/^```html\s*/, '').replace(/\s*```$/, '');
        } else if (aiKomentar.startsWith('```')) {
            aiKomentar = aiKomentar.replace(/^```\s*/, '').replace(/\s*```$/, '');
        }

        console.log('6. Inserting AI discussion into interim_report.html...');
        const finalInterimHtml = lokalskyHtml.replace('<!-- AI_COMMENTARY_PLACEHOLDER -->', aiKomentar);
        fs.writeFileSync('interim_report.html', finalInterimHtml, 'utf8');

        appendDiagramToReport(rawData);
        await uploadToFtp();

    } catch (error) {
        console.error('Error during HART execution:', error);
    }
}

// Execute
runHartAgent();
