import { Router } from 'express';
import { getArtifactsRoot } from '../db.js';

const router = Router();

router.get('/health', (req, res) => {
  res.json({ status: 'ok', artifacts_root: getArtifactsRoot() });
});

export default router;

