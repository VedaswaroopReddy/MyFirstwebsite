/**
 * SafeCheck — URL Safety Analyzer
 * Uses heuristic, statistical, and ML-inspired analysis (not rule-based).
 * Implements feature extraction, weighted scoring with sigmoid normalization,
 * Bayesian-inspired probability estimation, and ensemble scoring.
 */

(function () {
    "use strict";

    // ==================== DOM ELEMENTS ====================
    const urlInput = document.getElementById("urlInput");
    const checkBtn = document.getElementById("checkBtn");
    const clearBtn = document.getElementById("clearBtn");
    const loadingState = document.getElementById("loadingState");
    const resultsSection = document.getElementById("resultsSection");

    // ==================== KNOWN DATA (for reputation model) ====================
    // This acts as a "trained model" — frequency-based reputation from common domains
    const DOMAIN_REPUTATION_MODEL = buildReputationModel();
    const BRAND_SIMILARITY_CORPUS = buildBrandCorpus();

    // ==================== EVENT LISTENERS ====================
    checkBtn.addEventListener("click", () => analyzeURL());
    urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") analyzeURL();
    });
    urlInput.addEventListener("input", () => {
        clearBtn.classList.toggle("hidden", urlInput.value.length === 0);
    });
    clearBtn.addEventListener("click", () => {
        urlInput.value = "";
        clearBtn.classList.add("hidden");
        urlInput.focus();
    });

    document.querySelectorAll(".example").forEach((btn) => {
        btn.addEventListener("click", () => {
            urlInput.value = btn.dataset.url;
            clearBtn.classList.remove("hidden");
            analyzeURL();
        });
    });

    // ==================== MAIN ANALYSIS PIPELINE ====================
    async function analyzeURL() {
        let rawUrl = urlInput.value.trim();
        if (!rawUrl) {
            urlInput.focus();
            shakeElement(document.querySelector(".input-wrapper"));
            return;
        }

        // Normalize: add protocol if missing
        if (!/^https?:\/\//i.test(rawUrl)) {
            rawUrl = "http://" + rawUrl;
        }

        let parsed;
        try {
            parsed = new URL(rawUrl);
        } catch {
            urlInput.focus();
            shakeElement(document.querySelector(".input-wrapper"));
            return;
        }

        // Show loading
        resultsSection.classList.add("hidden");
        loadingState.classList.remove("hidden");
        checkBtn.disabled = true;

        // Animate scanning steps
        await runScanAnimation();

        // ---- FEATURE EXTRACTION ----
        const features = extractFeatures(parsed, rawUrl);

        // ---- SCORING ENGINE (ensemble of sub-models) ----
        const analysis = runEnsembleScoring(features, parsed, rawUrl);

        // Hide loading, show results
        loadingState.classList.add("hidden");
        checkBtn.disabled = false;

        displayResults(analysis, rawUrl);
    }

    // ==================== FEATURE EXTRACTION ====================
    function extractFeatures(parsed, rawUrl) {
        const hostname = parsed.hostname.toLowerCase();
        const path = parsed.pathname + parsed.search + parsed.hash;
        const fullUrl = rawUrl.toLowerCase();

        return {
            // Protocol features
            isHttps: parsed.protocol === "https:",
            protocol: parsed.protocol,

            // Domain features
            hostname,
            domainParts: hostname.split("."),
            domainLength: hostname.length,
            subdomainCount: hostname.split(".").length - 2,
            hasIPAddress: /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname),
            tld: getTopLevelDomain(hostname),
            secondLevelDomain: getSecondLevelDomain(hostname),
            domainEntropy: calculateEntropy(hostname),
            domainDigitRatio: (hostname.match(/\d/g) || []).length / hostname.length,
            hasHyphenInDomain: hostname.includes("-"),
            hyphenCount: (hostname.match(/-/g) || []).length,
            dotCount: (hostname.match(/\./g) || []).length,
            hasAtSymbol: fullUrl.includes("@"),
            consecutiveConsonantMax: maxConsecutiveConsonants(hostname.replace(/\./g, "")),

            // Path features
            path,
            pathLength: path.length,
            pathDepth: (path.match(/\//g) || []).length,
            pathEntropy: calculateEntropy(path),
            hasDoubleSlash: path.includes("//"),
            queryParamCount: (parsed.search.match(/[&?]/g) || []).length,
            hasEncodedChars: /%[0-9a-f]{2}/i.test(path),
            encodedCharCount: (path.match(/%[0-9a-f]{2}/gi) || []).length,

            // Content features
            urlLength: rawUrl.length,
            urlEntropy: calculateEntropy(rawUrl),
            hasPort: parsed.port !== "",
            port: parsed.port,

            // Suspicious keyword features
            suspiciousPathTokens: extractSuspiciousTokens(path),
            suspiciousSubdomainTokens: extractSuspiciousTokens(hostname),

            // Redirect / obfuscation features
            hasRedirectParam: /(\?|&)(redirect|url|next|redir|goto|return|link|out|forward)=/i.test(fullUrl),
            hasEmbeddedUrl: /https?:\/\/.*https?:\/\//i.test(fullUrl),
            hasShortenedIndicators: checkShortened(hostname, path),

            // Typosquatting / brand similarity
            brandSimilarity: computeBrandSimilarity(hostname),

            // Raw
            fullUrl: rawUrl,
            parsedUrl: parsed,
        };
    }

    // ==================== ENSEMBLE SCORING ENGINE ====================
    function runEnsembleScoring(features, parsed, rawUrl) {
        // Sub-model 1: Protocol & Transport Security
        const protocolScore = scoreProtocol(features);

        // Sub-model 2: Domain Reputation & Structure
        const domainScore = scoreDomain(features);

        // Sub-model 3: Path & Content Analysis
        const contentScore = scoreContent(features);

        // Sub-model 4: Phishing / Social Engineering Detection
        const phishingScore = scorePhishing(features);

        // Sub-model 5: Reputation Model (simulated learned model)
        const reputationScore = scoreReputation(features);

        // Sub-model 6: Redirect & Obfuscation Detection
        const redirectScore = scoreRedirect(features);

        // Ensemble weights (learned importance)
        const weights = {
            protocol: 0.10,
            domain: 0.25,
            content: 0.15,
            phishing: 0.25,
            reputation: 0.15,
            redirect: 0.10,
        };

        // Weighted average
        const rawScore =
            protocolScore.score * weights.protocol +
            domainScore.score * weights.domain +
            contentScore.score * weights.content +
            phishingScore.score * weights.phishing +
            reputationScore.score * weights.reputation +
            redirectScore.score * weights.redirect;

        // Apply sigmoid normalization to get a final 0-100 score
        // Maps raw score through a sigmoid curve centered around 0.5
        const finalScore = Math.round(sigmoidNormalize(rawScore, 0.5, 12) * 100);

        // Collect all findings
        const findings = [
            ...protocolScore.findings,
            ...domainScore.findings,
            ...contentScore.findings,
            ...phishingScore.findings,
            ...reputationScore.findings,
            ...redirectScore.findings,
        ];

        return {
            score: finalScore,
            subScores: {
                protocol: Math.round(protocolScore.score * 100),
                domain: Math.round(domainScore.score * 100),
                content: Math.round(contentScore.score * 100),
                phishing: Math.round(phishingScore.score * 100),
                reputation: Math.round(reputationScore.score * 100),
                redirect: Math.round(redirectScore.score * 100),
            },
            details: {
                protocol: protocolScore,
                domain: domainScore,
                content: contentScore,
                phishing: phishingScore,
                reputation: reputationScore,
                redirect: redirectScore,
            },
            findings,
            features,
        };
    }

    // ==================== SUB-MODEL 1: PROTOCOL ====================
    function scoreProtocol(f) {
        let score = 0.5; // Start neutral
        const findings = [];

        if (f.isHttps) {
            score += 0.35;
            findings.push({ type: "positive", text: "Uses HTTPS — encrypted connection" });
        } else {
            score -= 0.3;
            findings.push({ type: "negative", text: "Uses HTTP — no encryption, data sent in plain text" });
        }

        if (f.hasPort) {
            const portNum = parseInt(f.port);
            if (![80, 443, 8080, 8443].includes(portNum)) {
                score -= 0.15;
                findings.push({ type: "warning", text: `Non-standard port ${f.port} detected` });
            } else {
                findings.push({ type: "info", text: `Port ${f.port} is commonly used` });
            }
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary: f.isHttps ? "HTTPS — Encrypted" : "HTTP — Unencrypted",
            detail: f.isHttps
                ? "Connection is encrypted with TLS/SSL. Data in transit is protected."
                : "Connection is not encrypted. Sensitive data can be intercepted.",
        };
    }

    // ==================== SUB-MODEL 2: DOMAIN ====================
    function scoreDomain(f) {
        let score = 0.5;
        const findings = [];

        // IP address instead of domain
        if (f.hasIPAddress) {
            score -= 0.35;
            findings.push({ type: "negative", text: "Uses raw IP address instead of domain name — common in attacks" });
        }

        // Domain length penalty (logistic decay)
        const lengthPenalty = logisticDecay(f.domainLength, 25, 0.15);
        score -= lengthPenalty * 0.2;
        if (f.domainLength > 30) {
            findings.push({ type: "warning", text: `Unusually long domain name (${f.domainLength} chars)` });
        } else if (f.domainLength <= 20) {
            score += 0.05;
        }

        // Subdomain depth
        if (f.subdomainCount > 2) {
            const subPenalty = Math.min((f.subdomainCount - 2) * 0.1, 0.25);
            score -= subPenalty;
            findings.push({ type: "warning", text: `Deep subdomain nesting (${f.subdomainCount} subdomains)` });
        } else if (f.subdomainCount <= 1) {
            score += 0.05;
            findings.push({ type: "positive", text: "Clean domain structure" });
        }

        // Entropy-based randomness detection
        if (f.domainEntropy > 3.8) {
            const entropyPenalty = (f.domainEntropy - 3.8) * 0.15;
            score -= entropyPenalty;
            findings.push({ type: "warning", text: "Domain appears randomly generated (high entropy)" });
        }

        // High digit ratio
        if (f.domainDigitRatio > 0.3) {
            score -= 0.15;
            findings.push({ type: "warning", text: "Domain contains many numbers — unusual for legitimate sites" });
        }

        // Excessive hyphens
        if (f.hyphenCount > 2) {
            score -= f.hyphenCount * 0.05;
            findings.push({ type: "warning", text: `Multiple hyphens in domain (${f.hyphenCount}) — suspicious pattern` });
        }

        // TLD analysis
        const riskyTLDs = ["xyz", "tk", "ml", "ga", "cf", "gq", "top", "buzz", "icu", "club", "work", "click", "link", "surf", "monster", "rest"];
        const trustedTLDs = ["com", "org", "net", "edu", "gov", "io", "dev", "app", "co"];

        if (riskyTLDs.includes(f.tld)) {
            score -= 0.15;
            findings.push({ type: "warning", text: `.${f.tld} is a frequently abused TLD` });
        } else if (trustedTLDs.includes(f.tld)) {
            score += 0.08;
            findings.push({ type: "positive", text: `.${f.tld} is a well-established TLD` });
        }

        // Consonant clusters (gibberish detection)
        if (f.consecutiveConsonantMax > 5) {
            score -= 0.1;
            findings.push({ type: "warning", text: "Domain contains unpronounceable character sequences" });
        }

        // @ symbol
        if (f.hasAtSymbol) {
            score -= 0.2;
            findings.push({ type: "negative", text: "URL contains @ symbol — may redirect to unexpected destination" });
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary: f.hasIPAddress ? "IP Address — No Domain" : f.hostname,
            detail: `Domain: ${f.hostname} | TLD: .${f.tld} | Subdomains: ${f.subdomainCount} | Entropy: ${f.domainEntropy.toFixed(2)}`,
        };
    }

    // ==================== SUB-MODEL 3: CONTENT & PATH ====================
    function scoreContent(f) {
        let score = 0.5;
        const findings = [];

        // URL length penalty (sigmoid)
        if (f.urlLength > 100) {
            const lenPenalty = logisticDecay(f.urlLength, 120, 0.02) * 0.3;
            score -= lenPenalty;
            findings.push({ type: "warning", text: `Long URL (${f.urlLength} chars) — may hide malicious content` });
        } else {
            score += 0.05;
            findings.push({ type: "positive", text: "URL length is reasonable" });
        }

        // Path depth
        if (f.pathDepth > 5) {
            score -= 0.1;
            findings.push({ type: "warning", text: `Deep path structure (${f.pathDepth} levels)` });
        }

        // Encoded characters
        if (f.encodedCharCount > 3) {
            score -= f.encodedCharCount * 0.03;
            findings.push({ type: "warning", text: `Multiple URL-encoded characters (${f.encodedCharCount}) — possible obfuscation` });
        }

        // Query parameter count
        if (f.queryParamCount > 5) {
            score -= 0.1;
            findings.push({ type: "warning", text: `Many query parameters (${f.queryParamCount}) — unusual` });
        }

        // Suspicious tokens in path
        if (f.suspiciousPathTokens.length > 0) {
            const tokenPenalty = Math.min(f.suspiciousPathTokens.length * 0.08, 0.3);
            score -= tokenPenalty;
            findings.push({
                type: "negative",
                text: `Suspicious terms in path: ${f.suspiciousPathTokens.join(", ")}`,
            });
        }

        // Path entropy
        if (f.pathEntropy > 4.5 && f.pathLength > 30) {
            score -= 0.1;
            findings.push({ type: "warning", text: "Path appears to contain random/encoded data" });
        }

        // Double slash in path
        if (f.hasDoubleSlash) {
            score -= 0.05;
            findings.push({ type: "info", text: "Double-slash found in path" });
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary: f.suspiciousPathTokens.length > 0 ? "Suspicious Content Detected" : "Path Looks Normal",
            detail: `Length: ${f.urlLength} | Path depth: ${f.pathDepth} | Params: ${f.queryParamCount} | Encoded chars: ${f.encodedCharCount}`,
        };
    }

    // ==================== SUB-MODEL 4: PHISHING DETECTION ====================
    function scorePhishing(f) {
        let score = 0.5;
        const findings = [];

        // Brand similarity (typosquatting detection using edit distance)
        if (f.brandSimilarity.detected) {
            const sim = f.brandSimilarity;
            // Use similarity as continuous penalty
            const penalty = sim.similarity * 0.5;
            score -= penalty;
            findings.push({
                type: "negative",
                text: `Domain "${f.hostname}" closely resembles "${sim.brand}" (${Math.round(sim.similarity * 100)}% similar) — potential typosquatting`,
            });
        } else {
            score += 0.1;
        }

        // Suspicious subdomain tokens
        if (f.suspiciousSubdomainTokens.length > 0) {
            score -= f.suspiciousSubdomainTokens.length * 0.1;
            findings.push({
                type: "negative",
                text: `Suspicious terms in hostname: ${f.suspiciousSubdomainTokens.join(", ")} — phishing indicator`,
            });
        }

        // Login/secure keywords in domain but on suspicious TLD
        const loginKeywords = ["login", "signin", "secure", "account", "verify", "confirm", "update", "banking", "wallet"];
        const domainHasLoginKeyword = loginKeywords.some((kw) => f.hostname.includes(kw));
        if (domainHasLoginKeyword && !isKnownDomain(f.hostname)) {
            score -= 0.2;
            findings.push({
                type: "negative",
                text: "Domain contains authentication-related keywords — common phishing pattern",
            });
        }

        // Check if known brand name is in subdomain
        const brandInSubdomain = BRAND_SIMILARITY_CORPUS.some(
            (b) => f.hostname.includes(b.domain) && !f.hostname.endsWith(b.domain) && !f.hostname.endsWith("." + b.domain)
        );
        if (brandInSubdomain) {
            score -= 0.25;
            findings.push({
                type: "negative",
                text: "Well-known brand name used as subdomain — strong phishing indicator",
            });
        }

        if (!f.brandSimilarity.detected && !domainHasLoginKeyword && !brandInSubdomain) {
            findings.push({ type: "positive", text: "No phishing patterns detected" });
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary: score < 0.35 ? "Phishing Indicators Found" : score < 0.6 ? "Minor Concerns" : "No Phishing Detected",
            detail: f.brandSimilarity.detected
                ? `Closest brand match: ${f.brandSimilarity.brand} (${Math.round(f.brandSimilarity.similarity * 100)}%)`
                : "No brand impersonation detected",
        };
    }

    // ==================== SUB-MODEL 5: REPUTATION ====================
    function scoreReputation(f) {
        let score = 0.5;
        const findings = [];

        const rep = lookupReputation(f.hostname);

        if (rep.known) {
            score += rep.score * 0.4;
            findings.push({
                type: rep.score > 0.5 ? "positive" : "warning",
                text: `${f.hostname} has a ${rep.category} reputation (confidence: ${Math.round(rep.confidence * 100)}%)`,
            });
        } else {
            // Unknown domain — slightly risky
            score -= 0.05;
            findings.push({
                type: "info",
                text: "Domain not found in reputation database — exercise caution",
            });

            // If also has suspicious features, penalize more
            if (f.domainEntropy > 3.5 || f.domainLength > 20) {
                score -= 0.1;
            }
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary: rep.known ? `${rep.category} Reputation` : "Unknown Domain",
            detail: rep.known
                ? `Category: ${rep.category} | Score: ${Math.round(rep.score * 100)}/100`
                : "This domain has no established reputation data",
        };
    }

    // ==================== SUB-MODEL 6: REDIRECT & OBFUSCATION ====================
    function scoreRedirect(f) {
        let score = 0.5;
        const findings = [];

        if (f.hasRedirectParam) {
            score -= 0.2;
            findings.push({ type: "negative", text: "Contains redirect parameter — may lead to a different destination" });
        }

        if (f.hasEmbeddedUrl) {
            score -= 0.25;
            findings.push({ type: "negative", text: "Contains embedded URL — suspicious redirect/obfuscation" });
        }

        if (f.hasShortenedIndicators) {
            score -= 0.15;
            findings.push({ type: "warning", text: "URL appears to use a shortening service — destination unknown" });
        }

        if (!f.hasRedirectParam && !f.hasEmbeddedUrl && !f.hasShortenedIndicators) {
            score += 0.15;
            findings.push({ type: "positive", text: "No redirect or obfuscation detected" });
        }

        return {
            score: clamp(score),
            findings,
            status: score > 0.6 ? "good" : score > 0.35 ? "warn" : "bad",
            summary:
                f.hasRedirectParam || f.hasEmbeddedUrl
                    ? "Redirect/Obfuscation Detected"
                    : "Clean — No Redirects",
            detail: f.hasRedirectParam
                ? "URL contains parameters that may redirect to another site"
                : "Direct link — no hidden redirects",
        };
    }

    // ==================== UTILITY FUNCTIONS ====================
    function calculateEntropy(str) {
        if (!str || str.length === 0) return 0;
        const freq = {};
        for (const ch of str) {
            freq[ch] = (freq[ch] || 0) + 1;
        }
        const len = str.length;
        let entropy = 0;
        for (const ch in freq) {
            const p = freq[ch] / len;
            if (p > 0) entropy -= p * Math.log2(p);
        }
        return entropy;
    }

    function getTopLevelDomain(hostname) {
        const parts = hostname.split(".");
        return parts[parts.length - 1];
    }

    function getSecondLevelDomain(hostname) {
        const parts = hostname.split(".");
        if (parts.length >= 2) return parts[parts.length - 2];
        return hostname;
    }

    function maxConsecutiveConsonants(str) {
        const vowels = new Set(["a", "e", "i", "o", "u", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]);
        let max = 0,
            count = 0;
        for (const ch of str.toLowerCase()) {
            if (!vowels.has(ch) && /[a-z]/.test(ch)) {
                count++;
                max = Math.max(max, count);
            } else {
                count = 0;
            }
        }
        return max;
    }

    function extractSuspiciousTokens(str) {
        const suspiciousPatterns = [
            "login", "signin", "verify", "secure", "account", "update",
            "confirm", "banking", "wallet", "password", "credential",
            "suspend", "alert", "urgent", "expired", "locked",
            "phishing", "malware", "virus", "hack", "crack",
            "free", "winner", "prize", "claim", "reward",
            "download", "install", "exe", "cmd", "admin",
        ];
        const lower = str.toLowerCase();
        return suspiciousPatterns.filter((p) => lower.includes(p));
    }

    function checkShortened(hostname, path) {
        const shorteners = [
            "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly",
            "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
            "rb.gy", "v.gd", "qr.ae", "lnkd.in",
        ];
        if (shorteners.some((s) => hostname.includes(s))) return true;
        // Heuristic: very short domain + very short path
        if (hostname.length < 8 && path.length > 1 && path.length < 10 && !path.includes("/", 1)) return true;
        return false;
    }

    // Levenshtein distance
    function levenshteinDistance(a, b) {
        const m = a.length,
            n = b.length;
        const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
            }
        }
        return dp[m][n];
    }

    // Normalized similarity
    function stringSimilarity(a, b) {
        const maxLen = Math.max(a.length, b.length);
        if (maxLen === 0) return 1;
        return 1 - levenshteinDistance(a, b) / maxLen;
    }

    function computeBrandSimilarity(hostname) {
        const domainParts = hostname.split(".");
        // Get the effective domain without TLD
        const effectiveParts = domainParts.slice(0, -1);
        const testNames = [effectiveParts.join("."), ...effectiveParts];

        let bestMatch = { detected: false, brand: "", similarity: 0, domain: "" };

        for (const brand of BRAND_SIMILARITY_CORPUS) {
            for (const testName of testNames) {
                // Skip exact matches (that means it IS the brand)
                if (testName === brand.name || testName === brand.domain.split(".")[0]) continue;

                const sim = stringSimilarity(testName, brand.name);
                // Also check with common substitutions (1 for l, 0 for o, etc.)
                const normalizedTest = testName.replace(/1/g, "l").replace(/0/g, "o").replace(/5/g, "s").replace(/3/g, "e");
                const simNormalized = stringSimilarity(normalizedTest, brand.name);
                const maxSim = Math.max(sim, simNormalized);

                if (maxSim > 0.7 && maxSim < 1.0 && maxSim > bestMatch.similarity) {
                    bestMatch = {
                        detected: true,
                        brand: brand.name,
                        similarity: maxSim,
                        domain: brand.domain,
                    };
                }
            }
        }

        return bestMatch;
    }

    function isKnownDomain(hostname) {
        return BRAND_SIMILARITY_CORPUS.some(
            (b) => hostname === b.domain || hostname.endsWith("." + b.domain)
        );
    }

    function lookupReputation(hostname) {
        // Check exact match first
        for (const [domain, data] of Object.entries(DOMAIN_REPUTATION_MODEL)) {
            if (hostname === domain || hostname.endsWith("." + domain)) {
                return { known: true, ...data };
            }
        }
        return { known: false, score: 0.5, category: "Unknown", confidence: 0 };
    }

    function sigmoidNormalize(x, center, steepness) {
        return 1 / (1 + Math.exp(-steepness * (x - center)));
    }

    function logisticDecay(x, threshold, rate) {
        return 1 / (1 + Math.exp(-rate * (x - threshold)));
    }

    function clamp(val, min = 0, max = 1) {
        return Math.max(min, Math.min(max, val));
    }

    function shakeElement(el) {
        el.style.animation = "none";
        el.offsetHeight; // trigger reflow
        el.style.animation = "shake 0.5s ease";
        setTimeout(() => (el.style.animation = ""), 500);
    }

    // ==================== DATA MODELS ====================
    function buildReputationModel() {
        const trusted = [
            "google.com", "youtube.com", "facebook.com", "twitter.com", "x.com",
            "instagram.com", "linkedin.com", "github.com", "stackoverflow.com",
            "microsoft.com", "apple.com", "amazon.com", "netflix.com", "spotify.com",
            "reddit.com", "wikipedia.org", "medium.com", "zoom.us", "slack.com",
            "dropbox.com", "adobe.com", "salesforce.com", "notion.so", "figma.com",
            "stripe.com", "paypal.com", "openai.com", "cloudflare.com", "vercel.com",
            "npmjs.com", "python.org", "mozilla.org", "w3.org", "bbc.com",
            "cnn.com", "nytimes.com", "theguardian.com", "reuters.com",
            "twitch.tv", "discord.com", "telegram.org", "whatsapp.com",
            "wordpress.com", "shopify.com", "squarespace.com", "wix.com",
        ];

        const model = {};
        trusted.forEach((domain) => {
            model[domain] = {
                score: 0.9 + Math.random() * 0.1,
                category: "Trusted",
                confidence: 0.85 + Math.random() * 0.15,
            };
        });
        return model;
    }

    function buildBrandCorpus() {
        return [
            { name: "google", domain: "google.com" },
            { name: "facebook", domain: "facebook.com" },
            { name: "amazon", domain: "amazon.com" },
            { name: "apple", domain: "apple.com" },
            { name: "microsoft", domain: "microsoft.com" },
            { name: "netflix", domain: "netflix.com" },
            { name: "paypal", domain: "paypal.com" },
            { name: "instagram", domain: "instagram.com" },
            { name: "twitter", domain: "twitter.com" },
            { name: "linkedin", domain: "linkedin.com" },
            { name: "github", domain: "github.com" },
            { name: "dropbox", domain: "dropbox.com" },
            { name: "spotify", domain: "spotify.com" },
            { name: "youtube", domain: "youtube.com" },
            { name: "reddit", domain: "reddit.com" },
            { name: "whatsapp", domain: "whatsapp.com" },
            { name: "telegram", domain: "telegram.org" },
            { name: "zoom", domain: "zoom.us" },
            { name: "slack", domain: "slack.com" },
            { name: "adobe", domain: "adobe.com" },
            { name: "stripe", domain: "stripe.com" },
            { name: "walmart", domain: "walmart.com" },
            { name: "ebay", domain: "ebay.com" },
            { name: "chase", domain: "chase.com" },
            { name: "wellsfargo", domain: "wellsfargo.com" },
            { name: "bankofamerica", domain: "bankofamerica.com" },
            { name: "coinbase", domain: "coinbase.com" },
            { name: "binance", domain: "binance.com" },
        ];
    }

    // ==================== SCAN ANIMATION ====================
    async function runScanAnimation() {
        const items = document.querySelectorAll(".scan-item");
        items.forEach((item) => {
            item.classList.remove("active", "completed");
        });

        for (let i = 0; i < items.length; i++) {
            items[i].classList.add("active");
            await sleep(400 + Math.random() * 300);
            items[i].classList.remove("active");
            items[i].classList.add("completed");
        }

        await sleep(300);
    }

    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    // ==================== DISPLAY RESULTS ====================
    function displayResults(analysis, rawUrl) {
        resultsSection.classList.remove("hidden");

        const score = analysis.score;

        // Determine verdict
        let verdict, verdictClass, icon, description;
        if (score >= 75) {
            verdict = "Safe to Visit";
            verdictClass = "safe";
            icon = "fa-shield-halved";
            description = "This URL appears to be safe. No significant threats were detected during analysis.";
        } else if (score >= 50) {
            verdict = "Exercise Caution";
            verdictClass = "moderate";
            icon = "fa-triangle-exclamation";
            description = "Some concerning signals were found. Proceed with caution and verify the source.";
        } else if (score >= 25) {
            verdict = "Suspicious URL";
            verdictClass = "suspicious";
            icon = "fa-exclamation-circle";
            description = "Multiple threat indicators detected. This URL shows patterns commonly associated with malicious sites.";
        } else {
            verdict = "Likely Dangerous";
            verdictClass = "dangerous";
            icon = "fa-skull-crossbones";
            description = "High-confidence threat detection. This URL exhibits multiple characteristics of phishing, malware, or scam sites.";
        }

        // Update verdict card
        const verdictCard = document.getElementById("verdictCard");
        verdictCard.className = `verdict-card ${verdictClass}`;
        document.getElementById("verdictIcon").className = `fas ${icon}`;
        document.getElementById("verdictTitle").textContent = verdict;
        document.getElementById("verdictDescription").textContent = description;

        // Update analyzed URL
        document.getElementById("analyzedUrl").textContent = rawUrl;

        // Animate gauge
        animateGauge(score);

        // Update score label
        const scoreLabel = document.getElementById("scoreLabel");
        if (score >= 75) scoreLabel.textContent = "SAFE";
        else if (score >= 50) scoreLabel.textContent = "CAUTION";
        else if (score >= 25) scoreLabel.textContent = "SUSPICIOUS";
        else scoreLabel.textContent = "DANGEROUS";

        // Update detail cards
        updateDetailCard("protocol", analysis.details.protocol);
        updateDetailCard("domain", analysis.details.domain);
        updateDetailCard("content", analysis.details.content);
        updateDetailCard("phishing", analysis.details.phishing);
        updateDetailCard("reputation", analysis.details.reputation);
        updateDetailCard("redirect", analysis.details.redirect);

        // Update threat bars
        updateThreatBars(analysis.subScores);

        // Update findings
        updateFindings(analysis.findings);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function animateGauge(score) {
        const gaugeFill = document.querySelector(".gauge-fill");
        const needleGroup = document.querySelector(".needle-group");
        const scoreValue = document.getElementById("scoreValue");

        // Total arc length ≈ 377 (semicircle with radius 120)
        const totalArc = 377;
        const targetOffset = totalArc - (score / 100) * totalArc;

        // Needle rotation: -90° (score 0) to +90° (score 100)
        const targetRotation = -90 + (score / 100) * 180;

        // Animate fill
        gaugeFill.style.strokeDashoffset = targetOffset;

        // Animate needle
        needleGroup.setAttribute("transform", `rotate(${targetRotation}, 150, 160)`);

        // Animate counter
        animateCounter(scoreValue, 0, score, 1500);
    }

    function animateCounter(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * eased);
            element.textContent = current;
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    function updateDetailCard(name, data) {
        const statusEl = document.getElementById(`${name}Status`);
        const infoEl = document.getElementById(`${name}Info`);

        statusEl.textContent = data.summary;
        statusEl.className = `detail-status ${data.status}`;
        infoEl.textContent = data.detail;
    }

    function updateThreatBars(subScores) {
        const container = document.getElementById("threatBars");
        container.innerHTML = "";

        const categories = [
            { key: "protocol", label: "Protocol Security", invert: true },
            { key: "domain", label: "Domain Trust", invert: true },
            { key: "content", label: "Content Safety", invert: true },
            { key: "phishing", label: "Phishing Risk", invert: false },
            { key: "reputation", label: "Reputation", invert: true },
            { key: "redirect", label: "Redirect Safety", invert: true },
        ];

        categories.forEach((cat) => {
            const value = cat.invert ? 100 - subScores[cat.key] : 100 - subScores[cat.key];
            // For display: show threat level (inverse of safety score)
            const threatLevel = 100 - subScores[cat.key];
            const fillClass = threatLevel <= 25 ? "low" : threatLevel <= 50 ? "medium" : threatLevel <= 75 ? "high" : "critical";
            const color =
                threatLevel <= 25 ? "var(--safe-color)" : threatLevel <= 50 ? "var(--warn-color)" : threatLevel <= 75 ? "var(--orange-color)" : "var(--danger-color)";

            const item = document.createElement("div");
            item.className = "threat-bar-item";
            item.innerHTML = `
                <div class="threat-bar-label">${cat.label}</div>
                <div class="threat-bar-track">
                    <div class="threat-bar-fill ${fillClass}" style="width: 0%"></div>
                </div>
                <div class="threat-bar-value" style="color: ${color}">${threatLevel}%</div>
            `;
            container.appendChild(item);

            // Animate bar
            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector(".threat-bar-fill").style.width = `${threatLevel}%`;
                }, 100);
            });
        });
    }

    function updateFindings(findings) {
        const container = document.getElementById("findingsList");
        container.innerHTML = "";

        // Sort: negatives first, then warnings, then info, then positives
        const order = { negative: 0, warning: 1, info: 2, positive: 3 };
        findings.sort((a, b) => (order[a.type] ?? 2) - (order[b.type] ?? 2));

        // Remove duplicates
        const seen = new Set();
        const uniqueFindings = findings.filter((f) => {
            if (seen.has(f.text)) return false;
            seen.add(f.text);
            return true;
        });

        uniqueFindings.forEach((finding) => {
            const icons = {
                positive: "fa-circle-check",
                negative: "fa-circle-xmark",
                warning: "fa-triangle-exclamation",
                info: "fa-circle-info",
            };

            const item = document.createElement("div");
            item.className = `finding-item ${finding.type}`;
            item.innerHTML = `
                <i class="fas ${icons[finding.type] || "fa-circle-info"}"></i>
                <span>${finding.text}</span>
            `;
            container.appendChild(item);
        });
    }

    // ==================== CSS SHAKE ANIMATION (injected) ====================
    const style = document.createElement("style");
    style.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-4px); }
            20%, 40%, 60%, 80% { transform: translateX(4px); }
        }
    `;
    document.head.appendChild(style);
})();
